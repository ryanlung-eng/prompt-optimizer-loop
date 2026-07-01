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
import json
from typing import Dict, List, Tuple

import httpx
from tenacity import RetryError, retry, stop_after_attempt, wait_exponential

from .config import DatabricksConfig
from .synthetic_data import SyntheticInput

_MAX_CONCURRENT = 10  # Cap concurrent Databricks calls
_MAX_TURNS = 4         # KA round-trips per test case before giving up

# In real n8n, these three expressions are resolved by n8n's own expression
# engine before the LLM ever sees the prompt — the model never sees literal
# "{{ }}" syntax in production. We resolve the same three here so the eval
# harness matches what the KA actually receives at runtime.
_SYNTHETIC_SLACK_USER_ID = "U0EVAL0001"
_CONVERSATION_EXPR = "{{ $('Thread Formatter').item.json.conversation }}"
_USER_ID_EXPR = "{{ $('Slack Trigger').item.json.user }}"
_TIME_SAVED_EXPR = "{{ $('AI Agent').item.json.output.time_saved }}"

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
        .replace(_USER_ID_EXPR, _SYNTHETIC_SLACK_USER_ID)
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
            f"{config.workspace_url}/serving-endpoints/{config.generation_endpoint}/invocations"
        )
        self._headers = {
            "Authorization": f"Bearer {config.token}",
            "Content-Type": "application/json",
        }

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(min=2, max=10))
    async def _call(
        self,
        client: httpx.AsyncClient,
        endpoint_url: str,
        system_prompt: str,
        user_message: str,
        want_trace_id: bool = False,
    ) -> Tuple[str, str]:
        """Returns (content, trace_id). trace_id is "" unless want_trace_id and the
        endpoint actually returned one — see x-mlflow-return-trace-id below."""
        headers = dict(self._headers)
        if want_trace_id:
            headers["x-mlflow-return-trace-id"] = "true"

        resp = await client.post(
            endpoint_url,
            headers=headers,
            json={
                "input": [
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
        trace_id = body.get("metadata", {}).get("trace_id", "") if want_trace_id else ""

        if "choices" in body:
            return body["choices"][0]["message"]["content"], trace_id
        if "output" in body:
            for item in body["output"]:
                if item.get("type") == "message":
                    for block in item.get("content", []):
                        if block.get("type") in ("output_text", "text") and block.get("text"):
                            return block["text"], trace_id
        raise ValueError(
            f"Unrecognized response shape from {endpoint_url}: {json.dumps(body)[:1500]}"
        )

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
                client, self._endpoint_url, resolved, latest_user_turn, want_trace_id=True
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
