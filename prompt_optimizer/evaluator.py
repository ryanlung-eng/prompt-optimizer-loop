"""
Evaluates a set of prompts by calling Databricks directly
(bypasses Slack/n8n trigger complexity for tight eval loops).

Simulates the real multi-turn Slack conversation: if the KA asks a
clarifying question instead of outputting a workflow, a separate
"simulated user" call answers it consistently with the original
scenario, and the exchange continues (up to a turn budget) until the
KA either outputs JSON or we give up. This is deliberately NOT forced
structured output — the KA is free to keep asking if it genuinely
needs to; we only treat it as a failure if it never converges.
"""
import asyncio
import hashlib
import json
from typing import Dict, List, Tuple

import httpx
from tenacity import RetryError, retry, stop_after_attempt, wait_random_exponential

from .config import DatabricksConfig
from .synthetic_data import SyntheticInput

# Multi-turn conversations mean each concurrent "slot" can burst up to
# _MAX_TURNS sequential calls against the SAME KA endpoint, not just one —
# so this needs to be lower than it would for a single-shot design, to avoid
# tripping the endpoint's rate limit.
_MAX_CONCURRENT = 4
_MAX_TURNS = 4         # KA round-trips per test case before giving up

# In real n8n, these three expressions are resolved by n8n's own expression
# engine before the LLM ever sees the prompt — the model never sees literal
# "{{ }}" syntax in production. We resolve the same three here so the eval
# harness matches what the KA actually receives at runtime.
_CONVERSATION_EXPR = "{{ $('Thread Formatter').item.json.conversation }}"
_USER_ID_EXPR = "{{ $('Slack Trigger').item.json.user }}"
_TIME_SAVED_EXPR = "{{ $('AI Agent').item.json.output.time_saved }}"


def _synthetic_user_id(inp: SyntheticInput) -> str:
    """
    A distinct per-input synthetic Slack user ID, not a single shared constant.
    Concurrent test cases previously all claimed to be the exact same user
    (U0EVAL0001) — if the KA endpoint does any server-side session/context
    caching keyed by user ID, that could cross-wire concurrent conversations
    together. Deterministic (hash of the input text), not random, so results
    stay reproducible across runs.
    """
    digest = hashlib.sha1(inp.text.encode()).hexdigest()[:8].upper()
    return f"U0EVAL{digest}"

_SIMULATED_USER_SYSTEM = """\
You are role-playing as the person who sent the ORIGINAL request below, in an \
ongoing Slack conversation with an AI workflow-building assistant. Stay fully \
consistent with everything you already said — never contradict it.

ORIGINAL REQUEST (everything you already told the assistant):
{original_text}

The assistant just asked a follow-up question. Reply the way the ORIGINAL \
REQUESTER naturally would:
  - If the question asks about something already covered above, restate or \
point back to that detail — don't invent a different, contradictory answer.
  - If the question asks about something genuinely not covered above, invent \
one specific, plausible, consistent detail (a real-sounding value) — don't \
deflect or say "I don't know."
  - Keep it short and natural, like a real Slack reply (1-3 sentences).

Return ONLY the reply text, nothing else — no roleplay framing, no preamble."""


def _resolve_prompt(system_prompt: str, conversation: str, inp: SyntheticInput) -> str:
    return (
        system_prompt
        .replace(_CONVERSATION_EXPR, conversation)
        .replace(_USER_ID_EXPR, _synthetic_user_id(inp))
        .replace(_TIME_SAVED_EXPR, str(inp.time_saved_minutes))
    )


def _looks_like_json(text: str) -> bool:
    return text.strip().startswith("{")


class WorkflowEvaluator:
    """
    Calls the Databricks model serving endpoint directly with a given system
    prompt + synthetic user message, returning the model's raw response.
    This lets us evaluate prompt changes without triggering the full Slack workflow.
    """

    def __init__(self, config: DatabricksConfig):
        self._config = config
        self._endpoint_url = (
            f"{config.workspace_url}/serving-endpoints/{config.eval_endpoint}/invocations"
        )
        self._generation_url = (
            f"{config.workspace_url}/serving-endpoints/{config.fast_generation_endpoint}/invocations"
        )
        self._headers = {
            "Authorization": f"Bearer {config.token}",
            "Content-Type": "application/json",
        }

    # Rate limiting (429) needs much more room than a transient network blip —
    # jittered backoff so concurrent slots don't all retry in lockstep and
    # re-trip the same limit together.
    @retry(stop=stop_after_attempt(6), wait=wait_random_exponential(multiplier=1, min=4, max=60))
    async def _call(
        self,
        client: httpx.AsyncClient,
        endpoint_url: str,
        system_prompt: str,
        user_message: str,
        want_trace_id: bool = False,
        use_responses_api: bool = False,
    ) -> Tuple[str, str]:
        """
        Returns (content, trace_id). trace_id is "" unless want_trace_id and the
        endpoint actually returned one — see x-mlflow-return-trace-id below.

        use_responses_api selects the request wire format:
          - True  -> {"input": [...]}     — the KA endpoint (ka-bd1cb93b-endpoint),
                     a Databricks Responses-style API. Confirmed via its own 400
                     error when sent "messages" instead.
          - False -> {"messages": [...]}  — standard chat-completions, used by
                     fast_generation_endpoint/generation_endpoint/judge_endpoint
                     everywhere else in this codebase. Confirmed via this same
                     400-on-"input" error.
        Response parsing handles both "choices" (chat-completions) and "output"
        (Responses-style) shapes regardless of which format was sent, since
        that side was already verified working for both endpoints.
        """
        headers = dict(self._headers)
        if want_trace_id:
            headers["x-mlflow-return-trace-id"] = "true"

        payload_key = "input" if use_responses_api else "messages"
        resp = await client.post(
            endpoint_url,
            headers=headers,
            json={
                payload_key: [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_message},
                ],
                "max_tokens": 1500,
                "temperature": 0.3,
            },
            timeout=90,
        )
        if resp.status_code >= 400:
            raise ValueError(
                f"{resp.status_code} from {endpoint_url}: {resp.text[:1500]}"
            )
        body = resp.json()
        trace_id = (body.get("metadata") or {}).get("trace_id", "") if want_trace_id else ""

        # Every dict access below uses `... or {}`/`... or []` rather than
        # dict.get(key, default), since get()'s default only applies when the
        # key is ABSENT — a key present with an explicit null value (which we've
        # already seen from these endpoints, e.g. "metadata": null) still
        # returns None and crashes a chained .get()/subscript otherwise.
        content = None
        choices = body.get("choices") or []
        if choices and isinstance(choices[0], dict):
            content = (choices[0].get("message") or {}).get("content")
        else:
            for item in body.get("output") or []:
                if not isinstance(item, dict) or item.get("type") != "message":
                    continue
                for block in item.get("content") or []:
                    if isinstance(block, dict) and block.get("type") in ("output_text", "text") and block.get("text"):
                        content = block["text"]
                        break
                if content:
                    break

        if content is None:
            raise ValueError(
                f"Unrecognized response shape from {endpoint_url}: {json.dumps(body)[:1500]}"
            )
        if not content.strip():
            finish_reason = choices[0].get("finish_reason") if choices and isinstance(choices[0], dict) else None
            raise ValueError(
                f"Empty content from {endpoint_url}. finish_reason={finish_reason!r}. "
                f"Raw response: {json.dumps(body)[:1500]}"
            )
        return content, trace_id

    async def _simulate_user_reply(
        self, client: httpx.AsyncClient, original_text: str, ka_question: str
    ) -> str:
        system = _SIMULATED_USER_SYSTEM.format(original_text=original_text)
        content, _ = await self._call(client, self._generation_url, system, ka_question)
        return content.strip()

    async def _run_conversation(
        self, client: httpx.AsyncClient, system_prompt: str, inp: SyntheticInput
    ) -> Tuple[str, List[dict], str]:
        """
        Runs the KA up to _MAX_TURNS times, answering clarifying questions with
        a simulated user turn each time, until it outputs JSON or we give up.
        Returns (final_response, transcript, trace_id) where transcript is a
        list of {"role": "user"/"ka", "content": str} entries, and trace_id is
        the KA's own MLflow trace ID for the FINAL turn (for pulling the native
        Databricks judge/Assessment scores for that exact call).
        """
        conversation = inp.text
        latest_user_turn = inp.text
        transcript = [{"role": "user", "content": inp.text}]

        for turn in range(_MAX_TURNS):
            resolved = _resolve_prompt(system_prompt, conversation, inp)
            response, trace_id = await self._call(
                client, self._endpoint_url, resolved, latest_user_turn,
                want_trace_id=True, use_responses_api=True,
            )
            transcript.append({"role": "ka", "content": response})

            if _looks_like_json(response) or turn == _MAX_TURNS - 1:
                return response, transcript, trace_id

            reply = await self._simulate_user_reply(client, inp.text, response)
            transcript.append({"role": "user", "content": reply})
            conversation = f"{conversation}\n\nAssistant: {response}\n\nUser: {reply}"
            latest_user_turn = reply

        return response, transcript, trace_id  # pragma: no cover — loop always returns above

    async def run_batch(
        self,
        system_prompt: str,
        inputs: List[SyntheticInput],
    ) -> List[Tuple[SyntheticInput, str, List[dict], str]]:
        """
        Run all inputs against a single system prompt concurrently, simulating
        multi-turn conversations where the KA asks clarifying questions.
        Returns [(input, final_response, transcript, trace_id), …]
        """
        sem = asyncio.Semaphore(_MAX_CONCURRENT)

        async def bounded_call(inp: SyntheticInput) -> Tuple[SyntheticInput, str, List[dict], str]:
            async with sem:
                async with httpx.AsyncClient() as client:
                    try:
                        response, transcript, trace_id = await self._run_conversation(
                            client, system_prompt, inp
                        )
                    except Exception as e:
                        cause = e.last_attempt.exception() if isinstance(e, RetryError) else e
                        print(f"  Warning: eval failed for '{inp.text[:60]}…': {cause}")
                        response, transcript, trace_id = f"[ERROR: {cause}]", [], ""
                    return inp, response, transcript, trace_id

        return await asyncio.gather(*[bounded_call(inp) for inp in inputs])

    async def run_multi_prompt_batch(
        self,
        node_prompts: Dict[str, str],
        inputs: List[SyntheticInput],
    ) -> Dict[str, List[Tuple[SyntheticInput, str, List[dict], str]]]:
        """
        Evaluate multiple nodes' prompts simultaneously.
        Returns {node_name: [(input, response, transcript, trace_id), …]}
        """
        tasks = {
            node_name: self.run_batch(prompt, inputs)
            for node_name, prompt in node_prompts.items()
        }
        results = await asyncio.gather(*tasks.values())
        return dict(zip(tasks.keys(), results))
