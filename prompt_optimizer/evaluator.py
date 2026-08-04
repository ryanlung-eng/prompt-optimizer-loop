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
from pathlib import Path
from typing import Awaitable, Callable, Dict, List, Optional, Tuple

import httpx
from tenacity import RetryError, retry, stop_after_attempt, wait_random_exponential

from . import execution_checker as _execution_checker_module
from . import rag_retriever as _rag_retriever_module
from . import validator as _validator_module
from .config import DatabricksConfig
from .synthetic_data import SyntheticInput
from .validator import validate_workflow_json

# Hashes THIS file's own source, validator.py's, check_params.js's,
# rag_retriever.py's, AND execution_checker.py's — all directly determine
# how a conversation plays out (self-repair trigger logic, what counts as
# "valid", and — for the custom-RAG arm — what retrieve_context() actually
# fetches: top_k, max_context_chars, query_type all live in rag_retriever.py,
# not just the live index content). Baked into the cache key below so any
# change here automatically invalidates stale cached conversations, instead
# of relying on remembering to bump a version number or clear the cache by
# hand.
# check_params.js is invoked via subprocess rather than imported, so it's
# easy to forget — exactly what happened here: three straight hallucination-
# detection bug fixes to it never invalidated the cache at all, because only
# the two .py files were being hashed at the time. Same class of bug nearly
# repeated with rag_retriever.py — a top_k/budget tuning change alone doesn't
# necessarily change the resolved system_prompt text enough to guarantee a
# cache miss, so it needs to be hashed explicitly too, not left to chance.
_LOGIC_VERSION = hashlib.sha256(
    (
        Path(__file__).read_text()
        + Path(_validator_module.__file__).read_text()
        + _validator_module._SCHEMA_CHECK_SCRIPT.read_text()
        + Path(_rag_retriever_module.__file__).read_text()
        + Path(_execution_checker_module.__file__).read_text()
    ).encode()
).hexdigest()[:16]

# Multi-turn conversations mean each concurrent "slot" can burst up to
# _MAX_TURNS sequential calls against the SAME KA endpoint, not just one —
# so this needs to be lower than it would for a single-shot design, to avoid
# tripping the endpoint's rate limit.
_MAX_CONCURRENT = 4
_MAX_TURNS = 7         # KA round-trips per test case before giving up — gives
                       # the self-repair loop below room to actually work, not
                       # just room to escape a clarifying-question stall.

# In real n8n, these three expressions are resolved by n8n's own expression
# engine before the LLM ever sees the prompt — the model never sees literal
# "{{ }}" syntax in production. We resolve the same three here so the eval
# harness matches what the KA actually receives at runtime.
_CONVERSATION_EXPR = "{{ $('Thread Formatter').item.json.conversation }}"
_USER_ID_EXPR = "{{ $('Slack Trigger').item.json.user }}"
_TIME_SAVED_EXPR = "{{ $('AI Agent').item.json.output.time_saved }}"

# Terse gateway/throttling phrases seen wrapped in an HTTP 200 body instead of
# a proper 429 — checked case-insensitively as whole phrases, not single loose
# words like "busy"/"overloaded" alone, since a genuine KA response can
# legitimately discuss rate limiting in passing when building an HTTP Request
# node with retry/batching options (this domain's own KB documents that
# option). Requires the content to ALSO be short — a real workflow-builder
# reply (JSON or even a short clarifying question) essentially never comes in
# this short; a gateway throttling notice almost always does.
_RATE_LIMIT_PHRASES = (
    "rate limit exceeded", "too many requests", "quota exceeded",
    "request was throttled", "please try again later",
    "server is currently overloaded", "server is currently unavailable",
)
_RATE_LIMIT_MAX_LEN = 200


def _looks_like_rate_limit_message(content: str) -> bool:
    if len(content) > _RATE_LIMIT_MAX_LEN:
        return False
    lowered = content.lower()
    return any(phrase in lowered for phrase in _RATE_LIMIT_PHRASES)


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


_SEAM = "\n\n---\n\n"


def _assemble_custom_rag_prompt(base_instructions: str, retrieved_block: str) -> str:
    """
    Shared by run_batch_custom_rag and run_batch_custom_rag_v2 — assembles
    the final system prompt from base_instructions + a retrieved-context
    block, handling both shapes base_instructions can take:

      - a fully-formed prompt that already embeds _CONVERSATION_EXPR/
        _USER_ID_EXPR/_TIME_SAVED_EXPR inline and has its own "\\n\\n---\\n\\n"
        seam (e.g. config.yaml's production prompt, used as-is to make an arm
        directly comparable to production) — retrieved_block is spliced in
        at the first such seam, so the placeholders aren't duplicated.
      - a frozen rules doc with no placeholders of its own (e.g.
        instructions.md) — retrieved_block + the Conversation/User ID/
        Minutes Saved trailer are appended after it.
    """
    if _CONVERSATION_EXPR in base_instructions and _SEAM in base_instructions:
        seam_idx = base_instructions.index(_SEAM)
        return (
            base_instructions[:seam_idx] + _SEAM + retrieved_block + _SEAM
            + base_instructions[seam_idx + len(_SEAM):]
        )
    return (
        f"{base_instructions}\n\n---\n\n{retrieved_block}"
        f"\n\n---\n\nConversation:\n{_CONVERSATION_EXPR}\n\n"
        f"User ID:\n{_USER_ID_EXPR}\n"
        f"(This is the Slack ID of the person you are building this workflow "
        f"for. If any outbound send needs an approval-DM recipient, use this "
        f"ID directly. Never ask the user for their own Slack ID or handle — "
        f"you already have it.)\n\n"
        f"Minutes Saved:\n{_TIME_SAVED_EXPR}"
    )




class WorkflowEvaluator:
    """
    Calls the Databricks model serving endpoint directly with a given system
    prompt + synthetic user message, returning the model's raw response.
    This lets us evaluate prompt changes without triggering the full Slack workflow.
    """

    def __init__(self, config: DatabricksConfig, cache_dir: Optional[str] = None):
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

        # With temperature=0 everywhere, the entire conversation is
        # deterministic for a given (prompt, input) pair — no reason to
        # re-spend tokens re-running an identical conversation across
        # multiple notebook sessions (e.g. iterating on judge.py/validator.py
        # without touching the prompt itself, which is most of this session).
        self._cache_path = Path(cache_dir) / "conversation_cache.json" if cache_dir else None
        self._cache: Dict[str, dict] = {}
        if self._cache_path and self._cache_path.exists():
            try:
                self._cache = json.loads(self._cache_path.read_text())
            except Exception as e:
                print(f"  Warning: could not load conversation cache: {e}")

    @staticmethod
    def _cache_key(system_prompt: str, inp: SyntheticInput, endpoint_url: str = "") -> str:
        # system_prompt + inp.text + inp.time_saved_minutes are the only three
        # *inputs* that feed into the resolved prompt / conversation trajectory
        # (see _resolve_prompt) — together they fully determine the outcome
        # at temperature=0. _LOGIC_VERSION is included too, since a code change
        # to how the conversation is conducted (this file or validator.py) is
        # just as capable of changing the outcome as a prompt change is — a
        # bare (prompt, input) key would silently keep serving conversations
        # generated under old, since-fixed conversation-handling logic.
        # endpoint_url is included too — the benchmark mode can send the
        # exact same prompt text to two different endpoints (raw Sonnet vs.
        # the production KA) on purpose, to isolate the endpoint as the only
        # variable; without this, the second arm would wrongly reuse the
        # first arm's cached response since the rest of the key is identical.
        raw = f"{system_prompt}:::{inp.text}:::{inp.time_saved_minutes}:::{_LOGIC_VERSION}:::{endpoint_url}"
        return hashlib.sha256(raw.encode()).hexdigest()

    def _save_cache(self) -> None:
        if not self._cache_path:
            return
        self._cache_path.parent.mkdir(parents=True, exist_ok=True)
        self._cache_path.write_text(json.dumps(self._cache, indent=2))

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
        use_responses_api: bool = False,
        max_tokens: int = 1500,
        temperature: float = 0.0,
    ) -> str:
        """
        use_responses_api selects the request wire format:
          - True  -> {"input": [{"role": "user", "content": system_prompt}]}
                     — the KA endpoint (ka-bd1cb93b-endpoint / "kbqa_agent").
                     Confirmed via a real production trace: this agent sends
                     the ENTIRE resolved prompt (instructions + conversation)
                     as a single "user"-role message — it does not use a
                     separate "system" role at all. user_message is ignored
                     for this branch; the caller must fully resolve
                     everything into system_prompt beforehand.
          - False -> {"messages": [{"role": "system", ...}, {"role": "user", ...}]}
                     — standard chat-completions, used by
                     fast_generation_endpoint/generation_endpoint/judge_endpoint
                     everywhere else in this codebase.
        Response parsing handles both "choices" (chat-completions) and "output"
        (Responses-style) shapes regardless of which format was sent, since
        that side was already verified working for both endpoints.

        Note: the KA endpoint's full raw response has been confirmed to never
        include a trace_id (model: "kbqa_agent" doesn't support
        x-mlflow-return-trace-id), so we don't request or parse one here.
        """
        if use_responses_api:
            payload_key, messages = "input", [{"role": "user", "content": system_prompt}]
        elif not user_message:
            # Benchmark mode: mirror the KA endpoint's own structure (the
            # entire resolved prompt as one user-role message) but through
            # the classic chat-completions format, so a raw model endpoint
            # gets an identical message shape to what the KA receives —
            # isolating the endpoint/knowledge-access as the only variable,
            # not also the system/user role split.
            payload_key, messages = "messages", [{"role": "user", "content": system_prompt}]
        else:
            payload_key = "messages"
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_message},
            ]
        resp = await client.post(
            endpoint_url,
            headers=self._headers,
            json={
                payload_key: messages,
                "max_tokens": max_tokens,
                "temperature": temperature,
            },
            timeout=90,
        )
        if resp.status_code >= 400:
            raise ValueError(
                f"{resp.status_code} from {endpoint_url}: {resp.text[:1500]}"
            )
        body = resp.json()

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
        # Some gateways/serving layers throttle by returning HTTP 200 with a
        # rate-limit/throttling message AS the message content, rather than a
        # proper 4xx status — the status-code check above never sees this, so
        # it was previously accepted as a genuine (if content-free) KA
        # response, cached, and later misread by the gap report as "the KA
        # doesn't know this integration" rather than "this call got throttled."
        # Treat it as retryable instead, same as an explicit 429 would be.
        if _looks_like_rate_limit_message(content):
            raise ValueError(
                f"Suspected rate-limit/throttling response disguised as 200 OK from "
                f"{endpoint_url}: {content[:300]}"
            )
        return content

    async def _simulate_user_reply(
        self, client: httpx.AsyncClient, original_text: str, ka_question: str
    ) -> str:
        system = _SIMULATED_USER_SYSTEM.format(original_text=original_text)
        # temperature=0: during prompt isolation/optimization, every source of
        # non-determinism in the pipeline adds noise to prompt-vs-prompt
        # comparisons. Revisit as a hyperparameter once baseline prompt
        # performance is established.
        content = await self._call(client, self._generation_url, system, ka_question, temperature=0.0)
        return content.strip()

    async def _run_conversation(
        self, client: httpx.AsyncClient, system_prompt: str, inp: SyntheticInput,
        endpoint_url: Optional[str] = None, use_responses_api: bool = True,
        retrieve_on_repair: Optional[Callable[[str], Awaitable[str]]] = None,
        execution_checker: Optional[Callable[[str], Awaitable[List[str]]]] = None,
    ) -> Tuple[str, List[dict]]:
        """
        Runs the KA up to _MAX_TURNS times, answering clarifying questions with
        a simulated user turn each time, until it outputs JSON or we give up.
        Returns (final_response, transcript) where transcript is a list of
        {"role": "user"/"ka", "content": str} entries for debugging/logging.

        Confirmed against a real production trace: this agent expects the
        FULL running conversation (every turn, "User:"/"Assistant:" prefixed,
        including the very first message) embedded in a single resolved
        prompt sent as one "user"-role message — not split across separate
        system/user turns.

        endpoint_url/use_responses_api default to the production KA endpoint
        (self._endpoint_url, Responses API) — override for benchmark.py to
        run the exact same self-repair loop against a raw model endpoint
        instead, so the two are compared on identical conversation-handling
        logic, not two different code paths.

        Self-repair: more turns alone only helps CONVERGENCE (escaping a
        clarifying-question stall) — it does nothing for the CORRECTNESS of
        the JSON once produced. So once a response looks JSON-shaped, it's
        additionally checked against validate_workflow_json() before being
        accepted as final. If it fails, the specific validation errors are
        fed back as the next turn ("that didn't parse — here's why — please
        fix it"), giving the model a targeted chance to correct itself
        instead of us silently accepting broken JSON as the end result.

        retrieve_on_repair (custom-RAG arm only — None for production, so its
        behavior is unchanged): system_prompt's retrieved context is fixed at
        conversation start, computed from the ORIGINAL request text alone. A
        repair turn's error is often about a specific node/param that first
        query's top-K never happened to surface (e.g. request was about a
        Slack flow, but the actual bug is a Jira field) — with no new
        information entering the loop, the model just re-guesses against the
        same context for every remaining turn. When provided, this is called
        with the joined validation-error text on each repair turn, and its
        return value (a second, error-targeted retrieval) is appended to that
        turn's reply so the model has a real chance to fix the SPECIFIC thing
        that broke, not just retry blind.

        execution_checker (custom-RAG arm only — None elsewhere, so behavior
        is unchanged by default): called with a structurally-VALID response
        before it's accepted as final. Distinct from structural validation —
        this targets the "right node types, wrong execution behavior" class
        of bug (cross-node references that won't resolve, guards that don't
        cover every action, missing Merge synchronization) that passes every
        structural check but wouldn't actually work. Returns a list of issue
        strings; a non-empty list is fed back as a repair turn exactly like
        a structural error, consuming one turn of the shared budget.
        """
        endpoint_url = endpoint_url or self._endpoint_url
        conversation = f"User: {inp.text}"
        transcript = [{"role": "user", "content": inp.text}]

        for turn in range(_MAX_TURNS):
            resolved = _resolve_prompt(system_prompt, conversation, inp)
            # A complete workflow with an approval sub-flow can easily need
            # 8-10+ verbose n8n nodes — 1500 tokens (the old default, still
            # used for the short simulated-user replies) was truncating the
            # KA's output mid-generation, producing invalid JSON and
            # connections referencing nodes that never got emitted.
            # temperature=0: isolates prompt quality from sampling noise
            # during optimization — see _simulate_user_reply for the same rationale.
            response = await self._call(
                client, endpoint_url, resolved, "",
                use_responses_api=use_responses_api, max_tokens=6000, temperature=0.0,
            )
            transcript.append({"role": "ka", "content": response})
            last_turn = turn == _MAX_TURNS - 1

            # validate_workflow_json is the single source of truth for "is
            # this JSON" — a separate, stricter heuristic here previously
            # drifted out of sync with it (whole-string brace check vs. the
            # validator's find-anywhere check), causing a response to be
            # bucketed as "never attempted" in one place while still
            # generating a structural error in another.
            structural = validate_workflow_json(response)
            if structural.is_json and structural.valid:
                if last_turn or execution_checker is None:
                    return response, transcript
                # Structurally valid, but not yet accepted — give the
                # narrow execution-trace checker (see execution_checker.py)
                # a chance to catch the "right node types, wrong runtime
                # behavior" class of bug before this is returned as final.
                exec_issues = await execution_checker(response)
                if not exec_issues:
                    return response, transcript
                reply = (
                    f"That parses and passes structural checks, but tracing "
                    f"through its actual execution surfaced real problems: "
                    f"{'; '.join(exec_issues)}. Output ONLY the corrected "
                    f"workflow JSON — start your reply immediately with '{{' "
                    f"and end with '}}'. Do not diagnose the problem, explain "
                    f"your reasoning, or write any prose before or after the "
                    f"JSON; every token spent explaining is a token not spent "
                    f"on the workflow itself."
                )
            elif structural.is_json:
                if last_turn:
                    return response, transcript
                error_text = "; ".join(structural.errors)
                # "Can you fix it and send the corrected JSON?" left room for
                # the model to diagnose the error in prose first — observed
                # directly in real transcripts ("The core issue is that the
                # validator treats...", multiple paragraphs of reasoning
                # before the JSON even starts). That prose eats into the
                # fixed max_tokens=6000 completion budget, and several
                # "Invalid JSON: Expecting ',' delimiter" / "Extra data"
                # parse errors on repair turns are consistent with the JSON
                # itself getting cut off mid-generation as a result. Telling
                # the model explicitly to skip the diagnosis reclaims that
                # budget for the actual output.
                reply = (
                    f"That didn't work — I tried to import it and got these "
                    f"errors: {error_text}. Output ONLY the corrected "
                    f"workflow JSON — start your reply immediately with '{{' "
                    f"and end with '}}'. Do not diagnose the error, explain "
                    f"your reasoning, or write any prose before or after the "
                    f"JSON; every token spent explaining is a token not spent "
                    f"on the workflow itself."
                )
                if retrieve_on_repair is not None:
                    extra = await retrieve_on_repair(error_text)
                    if extra:
                        reply += (
                            "\n\nAdditional reference documentation that may "
                            f"help fix this specific error:\n\n{extra}"
                        )
            elif last_turn:
                return response, transcript
            else:
                reply = await self._simulate_user_reply(client, inp.text, response)

            transcript.append({"role": "user", "content": reply})
            conversation = f"{conversation}\n\nAssistant: {response}\n\nUser: {reply}"

        return response, transcript  # pragma: no cover — loop always returns above

    async def run_batch(
        self,
        system_prompt: str,
        inputs: List[SyntheticInput],
        endpoint_url: Optional[str] = None,
        use_responses_api: bool = True,
    ) -> List[Tuple[SyntheticInput, str, List[dict]]]:
        """
        Run all inputs against a single system prompt concurrently, simulating
        multi-turn conversations where the KA asks clarifying questions.
        Returns [(input, final_response, transcript), …]

        endpoint_url/use_responses_api default to the production KA endpoint —
        override for benchmark.py to run the identical self-repair loop
        against a raw model endpoint instead.
        """
        endpoint_url = endpoint_url or self._endpoint_url
        sem = asyncio.Semaphore(_MAX_CONCURRENT)
        cache_hits = 0

        async def bounded_call(inp: SyntheticInput) -> Tuple[SyntheticInput, str, List[dict]]:
            nonlocal cache_hits
            key = self._cache_key(system_prompt, inp, endpoint_url)
            cached = self._cache.get(key)
            if cached is not None:
                cache_hits += 1
                return inp, cached["response"], cached["transcript"]

            async with sem:
                async with httpx.AsyncClient() as client:
                    try:
                        response, transcript = await self._run_conversation(
                            client, system_prompt, inp,
                            endpoint_url=endpoint_url, use_responses_api=use_responses_api,
                        )
                    except Exception as e:
                        cause = e.last_attempt.exception() if isinstance(e, RetryError) else e
                        print(f"  Warning: eval failed for '{inp.text[:60]}…': {cause}")
                        response, transcript = f"[ERROR: {cause}]", []
                    # Don't cache errors — rate limits/backend outages are
                    # transient infra issues, not deterministic model output;
                    # a later run might succeed even with the same key.
                    if not response.startswith("[ERROR:"):
                        self._cache[key] = {"response": response, "transcript": transcript}
                        # Save after EVERY new entry, not just once at the end —
                        # a hard rate limit forcing an interrupted/killed run
                        # previously meant every conversation that DID succeed
                        # before the interruption was lost, since _save_cache()
                        # was never reached. Asyncio is cooperative/single-
                        # threaded, so this synchronous write can't race with
                        # another coroutine's cache mutation.
                        self._save_cache()
                    return inp, response, transcript

        results = await asyncio.gather(*[bounded_call(inp) for inp in inputs])
        if cache_hits:
            print(f"  {cache_hits}/{len(inputs)} conversations served from cache (0 tokens used)")
        self._save_cache()
        return results

    async def run_batch_custom_rag(
        self,
        base_instructions: str,
        inputs: List[SyntheticInput],
        rag_config,  # RAGConfig, from .rag_retriever — typed loosely to avoid
                     # a hard import dependency on databricks-ai-search for
                     # callers that never exercise this path.
        endpoint_url: Optional[str] = None,
        use_responses_api: bool = False,
    ) -> List[Tuple[SyntheticInput, str, List[dict]]]:
        """
        Like run_batch, but the system prompt is assembled PER INPUT instead
        of being one fixed string for the whole batch: base_instructions +
        retrieved context from rag_config's index for THIS input's own query
        text. Retrieval genuinely varies per request, so — unlike the rest of
        this pipeline — one shared prompt can't represent the custom-RAG arm.

        base_instructions can be either:
          - a frozen rules doc with no placeholders of its own (e.g.
            instructions.md) — retrieved context + the Conversation/User ID/
            Minutes Saved trailer are appended after it.
          - a fully-formed prompt that already embeds _CONVERSATION_EXPR/
            _USER_ID_EXPR/_TIME_SAVED_EXPR inline and has its own "\\n\\n---\\n\\n"
            seam (e.g. config.yaml's production prompt, used as-is to make an
            arm directly comparable to production — the ONLY difference from
            what production receives is the extra retrieved-context block
            spliced in at that seam) — retrieved context is spliced in at the
            first such seam instead, so the placeholders aren't duplicated.

        endpoint_url defaults to rag_config.generation_endpoint (raw Claude,
        chat-completions — Sonnet unless overridden in config.yaml, e.g. to
        try Opus) — this arm has no KA endpoint of its own, it's meant to be
        compared against one (see benchmark_rag_vs_ka.py). Previously
        defaulted to self._generation_url (fast_generation_endpoint / Haiku)
        by accident — every custom-RAG benchmark run before this fix was
        generating on Haiku, not a model comparable to production's.
        """
        from dataclasses import replace as _dc_replace

        from .rag_retriever import retrieve_context

        endpoint_url = endpoint_url or (
            f"{self._config.workspace_url}/serving-endpoints/"
            f"{rag_config.generation_endpoint}/invocations"
        )
        sem = asyncio.Semaphore(_MAX_CONCURRENT)
        cache_hits = 0

        # Repair-turn retrieval uses a smaller budget than the initial
        # retrieval — it's meant to fetch a couple of targeted chunks for one
        # specific validation error, not re-ground the whole request. Without
        # this cap, up to _MAX_TURNS-1 repair turns could each append a full
        # top-K context block to the ever-growing `conversation` string.
        _repair_rag_config = _dc_replace(rag_config, top_k=3, max_context_chars=4000)

        async def _retrieve_on_repair(error_text: str) -> str:
            # Belt-and-braces: an empty query makes Databricks AI Search raise
            # "Query text is required when reranking is enabled" and kills the
            # whole eval for that input. The root cause (a check that failed
            # .valid without appending to .errors) is fixed in validator.py,
            # but nothing downstream should be able to turn a caller's empty
            # string into a hard failure of the run.
            if not (error_text or "").strip():
                return ""
            return await asyncio.to_thread(retrieve_context, error_text, _repair_rag_config)

        async def bounded_call(inp: SyntheticInput) -> Tuple[SyntheticInput, str, List[dict]]:
            nonlocal cache_hits
            try:
                # retrieve_context() makes a blocking Databricks SDK call — run
                # it off the event loop so it doesn't stall other concurrent
                # slots. retrieve_chunks() itself retries transient Model
                # Serving routing failures (the embedding endpoint is re-hit on
                # every single call, not just at index-build time) — this
                # try/except is the backstop for when retries are exhausted,
                # so ONE scenario's retrieval failure produces an [ERROR: ...]
                # result for just that scenario instead of raising out of
                # bounded_call and crashing the entire asyncio.gather() batch.
                retrieved = await asyncio.to_thread(retrieve_context, inp.text, rag_config)
            except Exception as e:
                cause = e.last_attempt.exception() if isinstance(e, RetryError) else e
                print(f"  Warning: retrieval failed for '{inp.text[:60]}…': {cause}")
                return inp, f"[ERROR: retrieval failed: {cause}]", []
            retrieved_block = (
                f"Retrieved reference documentation (top-{rag_config.top_k} most "
                f"relevant sections for this request — use this as the authoritative "
                f"source for exact parameter names and node behavior beyond what's "
                f"already above):\n\n{retrieved}"
            )
            # _CONVERSATION_EXPR/_USER_ID_EXPR/_TIME_SAVED_EXPR must end up
            # present exactly once in the resolved template — _run_conversation's
            # self-repair loop calls _resolve_prompt() on every turn, which
            # substitutes these exact placeholder strings wherever they occur.
            system_prompt = _assemble_custom_rag_prompt(base_instructions, retrieved_block)

            key = self._cache_key(system_prompt, inp, endpoint_url)
            cached = self._cache.get(key)
            if cached is not None:
                cache_hits += 1
                return inp, cached["response"], cached["transcript"]

            async with sem:
                async with httpx.AsyncClient() as client:
                    try:
                        response, transcript = await self._run_conversation(
                            client, system_prompt, inp,
                            endpoint_url=endpoint_url, use_responses_api=use_responses_api,
                            retrieve_on_repair=_retrieve_on_repair,
                        )
                    except Exception as e:
                        cause = e.last_attempt.exception() if isinstance(e, RetryError) else e
                        print(f"  Warning: custom-RAG eval failed for '{inp.text[:60]}…': {cause}")
                        response, transcript = f"[ERROR: {cause}]", []
                    if not response.startswith("[ERROR:"):
                        self._cache[key] = {"response": response, "transcript": transcript}
                        self._save_cache()
                    return inp, response, transcript

        results = await asyncio.gather(*[bounded_call(inp) for inp in inputs])
        if cache_hits:
            print(f"  {cache_hits}/{len(inputs)} conversations served from cache (0 tokens used)")
        self._save_cache()
        return results

    async def run_batch_custom_rag_v2(
        self,
        base_instructions: str,
        inputs: List[SyntheticInput],
        rag_config,  # RAGConfig, from .rag_retriever — see run_batch_custom_rag
        endpoint_url: Optional[str] = None,
        use_responses_api: bool = False,
        execution_check: bool = False,
    ) -> List[Tuple[SyntheticInput, str, List[dict]]]:
        """
        Identical to run_batch_custom_rag except the initial retrieval is run
        through rag_pipeline_v2's relevance filter before being formatted into
        the prompt (see rag_pipeline_v2.py's module docstring for why this is
        its own arm instead of replacing run_batch_custom_rag outright).
        Repair-turn retrieval is intentionally UNFILTERED, same as v1 — it's
        already narrow (top_k=3, targeted at one specific validation error),
        so filtering it further has little to gain and one more LLM call to
        lose time/money on.

        execution_check=True wires in execution_checker.py's narrow
        pre-deployment execution-trace check (see that module's docstring) —
        a separately-benchmarkable toggle, not a silent change to this arm's
        existing measured behavior. The cache key includes the flag itself
        (not just this file's source): otherwise a checked and unchecked run
        against the SAME input/prompt/endpoint would collide on the same
        cache entry within one benchmark run and silently never exercise the
        checked path at all.
        """
        from dataclasses import replace as _dc_replace

        from .execution_checker import check_execution
        from .rag_pipeline_v2 import build_retrieved_block, retrieve_and_filter
        from .rag_retriever import retrieve_context

        endpoint_url = endpoint_url or (
            f"{self._config.workspace_url}/serving-endpoints/"
            f"{rag_config.generation_endpoint}/invocations"
        )
        # Relevance filter uses the same cheap/fast model as simulated-user
        # replies (self._generation_url = fast_generation_endpoint) — mirrors
        # HR Bot's own choice of a cheap model for classify/filter, reserving
        # the stronger model for generation only.
        filter_endpoint_url = self._generation_url
        filter_headers = self._headers

        sem = asyncio.Semaphore(_MAX_CONCURRENT)
        cache_hits = 0
        total_retrieved = 0
        total_kept = 0

        _repair_rag_config = _dc_replace(rag_config, top_k=3, max_context_chars=4000)

        async def _retrieve_on_repair(error_text: str) -> str:
            # Belt-and-braces: an empty query makes Databricks AI Search raise
            # "Query text is required when reranking is enabled" and kills the
            # whole eval for that input. The root cause (a check that failed
            # .valid without appending to .errors) is fixed in validator.py,
            # but nothing downstream should be able to turn a caller's empty
            # string into a hard failure of the run.
            if not (error_text or "").strip():
                return ""
            return await asyncio.to_thread(retrieve_context, error_text, _repair_rag_config)

        execution_checker = None
        if execution_check:
            async def execution_checker(workflow_json_text: str) -> List[str]:
                # Same cheap/fast model as the relevance filter — this is a
                # narrow, single-purpose check, not a reasoning-heavy review.
                async with httpx.AsyncClient() as check_client:
                    return await check_execution(
                        check_client, filter_endpoint_url, filter_headers, workflow_json_text,
                    )

        async def bounded_call(inp: SyntheticInput) -> Tuple[SyntheticInput, str, List[dict]]:
            nonlocal cache_hits, total_retrieved, total_kept
            try:
                # retrieve_chunks() (called inside retrieve_and_filter) retries
                # transient Model Serving routing failures itself — this
                # try/except is the backstop for when retries are exhausted,
                # so ONE scenario's retrieval failure produces an [ERROR: ...]
                # result for just that scenario instead of raising out of
                # bounded_call and crashing the entire asyncio.gather() batch.
                async with httpx.AsyncClient() as filter_client:
                    retrieved, kept = await retrieve_and_filter(
                        filter_client, filter_endpoint_url, filter_headers, inp.text, rag_config,
                    )
            except Exception as e:
                cause = e.last_attempt.exception() if isinstance(e, RetryError) else e
                print(f"  Warning: retrieval failed for '{inp.text[:60]}…': {cause}")
                return inp, f"[ERROR: retrieval failed: {cause}]", []
            total_retrieved += len(retrieved)
            total_kept += len(kept)
            retrieved_block = build_retrieved_block(kept, len(retrieved), rag_config)
            system_prompt = _assemble_custom_rag_prompt(base_instructions, retrieved_block)

            key = self._cache_key(system_prompt, inp, endpoint_url)
            if execution_check:
                key += ":::execcheck"
            cached = self._cache.get(key)
            if cached is not None:
                cache_hits += 1
                return inp, cached["response"], cached["transcript"]

            async with sem:
                async with httpx.AsyncClient() as client:
                    try:
                        response, transcript = await self._run_conversation(
                            client, system_prompt, inp,
                            endpoint_url=endpoint_url, use_responses_api=use_responses_api,
                            retrieve_on_repair=_retrieve_on_repair,
                            execution_checker=execution_checker,
                        )
                    except Exception as e:
                        cause = e.last_attempt.exception() if isinstance(e, RetryError) else e
                        print(f"  Warning: custom-RAG-v2 eval failed for '{inp.text[:60]}…': {cause}")
                        response, transcript = f"[ERROR: {cause}]", []
                    if not response.startswith("[ERROR:"):
                        self._cache[key] = {"response": response, "transcript": transcript}
                        self._save_cache()
                    return inp, response, transcript

        results = await asyncio.gather(*[bounded_call(inp) for inp in inputs])
        if cache_hits:
            print(f"  {cache_hits}/{len(inputs)} conversations served from cache (0 tokens used)")
        if total_retrieved:
            print(f"  Relevance filter kept {total_kept}/{total_retrieved} retrieved chunks "
                  f"({total_retrieved - total_kept} dropped) across {len(inputs)} inputs")
        self._save_cache()
        return results

    async def run_multi_prompt_batch(
        self,
        node_prompts: Dict[str, str],
        inputs: List[SyntheticInput],
    ) -> Dict[str, List[Tuple[SyntheticInput, str, List[dict]]]]:
        """
        Evaluate multiple nodes' prompts simultaneously.
        Returns {node_name: [(input, response, transcript), …]}
        """
        tasks = {
            node_name: self.run_batch(prompt, inputs)
            for node_name, prompt in node_prompts.items()
        }
        results = await asyncio.gather(*tasks.values())
        return dict(zip(tasks.keys(), results))
