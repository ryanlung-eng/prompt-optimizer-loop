"""
The custom-RAG workflow builder, packaged as ONE callable pipeline.

Everything the benchmark measures — retrieval, the relevance filter, the
grounding note, generation, structural validation and the self-repair loop —
runs inside a single `build()` call, so n8n makes one request and gets back
either a finished workflow or a clarifying question. n8n calls this via the
native Databricks node (resource: modelServing, operation: queryEndpoint),
which auto-detects the schema, so no HTTP Request plumbing is needed.

Why one endpoint rather than the current split (builder here, a separate
validator/debugger agent afterwards): the repair loop is only useful if the
thing that finds a defect can immediately hand it back to the thing that can
fix it, with the retrieved docs for that specific defect still in hand. Across
a service boundary that round trip is somebody else's orchestration problem;
inside one call it is a `for` loop. Consolidating also means the benchmark and
production run the SAME code path, so a benchmark number finally predicts
production behaviour instead of describing a harness nobody ships.

THE ONE DELIBERATE DIFFERENCE FROM THE BENCHMARK HARNESS
--------------------------------------------------------
In the benchmark, when the model replies with a clarifying question instead of
JSON, a simulated user answers it so the run can continue unattended. In
production a real person is on the other end, so that behaviour would be
actively wrong — it would invent an answer on the user's behalf and build
against it. Here a non-JSON reply is returned as `status="question"` for n8n
to relay. Repair turns (structural errors, platform-rule violations) are the
opposite case: those are the machine talking to itself about something it can
verify, so they stay internal and the user never sees them.
"""
import asyncio
import json
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional

import httpx

from .config import Config
from .evaluator import (
    _GATE_WARNING_MARKER,
    _REPAIRABLE_WARNING_MARKERS,
    _assemble_custom_rag_prompt,
    _has_explicit_gate_optout,
    _looks_like_truncated_json,
)
from .validator import validate_workflow_json

# Repair rounds allowed inside a single call. Lower than the benchmark's 7,
# because those 7 also had to absorb simulated-user clarification turns, which
# do not happen here — this budget is purely for machine-verifiable repairs.
# Each round is a full generation call, so this is the main latency lever.
MAX_REPAIR_ROUNDS = 3


@dataclass
class BuildResult:
    """Returned to n8n as the endpoint's response body."""
    status: str                       # "workflow" | "question" | "error"
    workflow: Optional[dict] = None   # parsed JSON, present when status="workflow"
    message: str = ""                 # the clarifying question, or the error text
    valid: bool = False
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    repair_rounds: int = 0
    kept_sources: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class WorkflowBuilderPipeline:
    """Retrieval + generation + validation + repair, behind one method.

    Constructed once per serving replica (loading config and warming the
    vector-search client is not per-request work), then `build()` per request.
    """

    def __init__(self, config: Config):
        self._config = config
        self._db = config.databricks
        self._rag = config.rag
        self._headers = {
            "Authorization": f"Bearer {self._db.token}",
            "Content-Type": "application/json",
        }
        self._generation_url = (
            f"{self._db.workspace_url}/serving-endpoints/"
            f"{self._rag.generation_endpoint}/invocations"
        )
        # The cheap model does the relevance filtering, mirroring the
        # benchmarked pipeline exactly — swapping it here would silently make
        # production a different system from the one that was measured.
        self._filter_url = (
            f"{self._db.workspace_url}/serving-endpoints/"
            f"{self._db.fast_generation_endpoint}/invocations"
        )

    # ---------------------------------------------------------------- call

    async def _generate(self, client: httpx.AsyncClient, prompt: str) -> str:
        payload: Dict[str, Any] = {"messages": [{"role": "user", "content": prompt}]}
        resp = await client.post(self._generation_url, headers=self._headers,
                                 json=payload, timeout=300)
        # Some endpoints (Opus 5) reject `temperature` outright; we do not send
        # it here at all, which sidesteps that entirely. Kept as a note because
        # the benchmark client DOES send it and had to learn this the hard way.
        if resp.status_code >= 400:
            raise RuntimeError(f"{resp.status_code} from generation endpoint: "
                               f"{resp.text[:500]}")
        body = resp.json()
        choices = body.get("choices") or []
        if choices and isinstance(choices[0], dict):
            content = (choices[0].get("message") or {}).get("content")
            if content:
                return content
        output = body.get("output")
        if isinstance(output, str):
            return output
        raise RuntimeError(f"Unrecognised generation response shape: "
                           f"{json.dumps(body)[:300]}")

    # ------------------------------------------------------------- repairs

    def _repairable_warnings(self, warnings: List[str], request_text: str) -> List[str]:
        """Platform-rule violations worth handing back for a fix.

        Gate warnings are dropped when the user explicitly asked for a send to
        be automatic — the static check cannot see intent, and re-gating an
        explicitly-declined send is an intent violation, not extra safety.
        """
        optout = _has_explicit_gate_optout(request_text)
        return [
            w for w in warnings
            if any(m in w for m in _REPAIRABLE_WARNING_MARKERS)
            and not (optout and _GATE_WARNING_MARKER in w)
        ]

    async def _retrieve_for(self, text: str) -> str:
        """Targeted re-retrieval for a specific defect, used to ground a repair
        turn — the same mechanism the benchmark uses, and the reason repairs
        fix things instead of inventing plausible-looking parameters."""
        from dataclasses import replace as _dc_replace

        from .rag_retriever import retrieve_context
        if not (text or "").strip():
            return ""
        cfg = _dc_replace(self._rag, top_k=3, max_context_chars=4000)
        try:
            return await asyncio.to_thread(retrieve_context, text, cfg)
        except Exception:
            return ""   # grounding is an enhancement; never fail the build for it

    # ----------------------------------------------------------------- api

    async def build_async(self, conversation: str) -> BuildResult:
        from .rag_pipeline_v2 import build_retrieved_block, retrieve_and_filter

        async with httpx.AsyncClient() as client:
            try:
                retrieved, kept = await retrieve_and_filter(
                    client, self._filter_url, self._headers, conversation, self._rag,
                )
            except Exception as e:
                return BuildResult(status="error", message=f"Retrieval failed: {e}")

            block = build_retrieved_block(kept, len(retrieved), self._rag)
            system_prompt = _assemble_custom_rag_prompt(
                self._config.prompts[self._config.benchmark.node_name], block,
            )
            kept_sources = sorted({c.source for c in kept})

            prompt = f"{system_prompt}\n\nUser: {conversation}"
            rounds = 0
            while True:
                try:
                    reply = await self._generate(client, prompt)
                except Exception as e:
                    return BuildResult(status="error", message=str(e),
                                       kept_sources=kept_sources, repair_rounds=rounds)

                structural = validate_workflow_json(reply)

                # Not JSON at all. In the harness a simulated user would answer
                # this; in production it belongs to the real user.
                if not structural.is_json:
                    if _looks_like_truncated_json(reply) and rounds < MAX_REPAIR_ROUNDS:
                        rounds += 1
                        prompt = (
                            f"{prompt}\n\nAssistant: {reply}\n\nUser: Your reply was "
                            f"cut off before the workflow JSON finished. Send the "
                            f"COMPLETE workflow JSON again in compact form — no "
                            f"indentation, no prose. Start with '{{' and end with '}}'."
                        )
                        continue
                    return BuildResult(status="question", message=reply.strip(),
                                       kept_sources=kept_sources, repair_rounds=rounds)

                repairable = self._repairable_warnings(structural.warnings, conversation)
                needs_fix = (not structural.valid) or repairable
                if not needs_fix or rounds >= MAX_REPAIR_ROUNDS:
                    return BuildResult(
                        status="workflow",
                        workflow=json.loads(
                            reply[reply.find("{"):reply.rfind("}") + 1]
                        ),
                        valid=structural.valid,
                        errors=structural.errors,
                        warnings=structural.warnings,
                        repair_rounds=rounds,
                        kept_sources=kept_sources,
                    )

                rounds += 1
                if not structural.valid:
                    detail = "; ".join(structural.errors)
                    instruction = (
                        f"That didn't work — I tried to import it and got these "
                        f"errors: {detail}."
                    )
                else:
                    detail = "; ".join(repairable)
                    instruction = (
                        f"That imports cleanly, but it violates required platform "
                        f"rules: {detail}. Fix these UNLESS my original request "
                        f"explicitly asked for the flagged behaviour."
                    )
                extra = await self._retrieve_for(detail)
                if extra:
                    instruction += (
                        f"\n\nReference documentation for this specific "
                        f"problem:\n\n{extra}"
                    )
                prompt = (
                    f"{prompt}\n\nAssistant: {reply}\n\nUser: {instruction} Output "
                    f"ONLY the corrected workflow JSON — start with '{{' and end "
                    f"with '}}'. No prose before or after."
                )

    def build(self, conversation: str) -> Dict[str, Any]:
        """Blocking entry point — what the MLflow model calls per request."""
        return asyncio.run(self.build_async(conversation)).to_dict()


def _conversation_from_payload(payload: Any) -> str:
    """Accepts the shapes n8n might send and normalises to one string.

    Deliberately permissive: `queryEndpoint` auto-detects the endpoint schema,
    and the calling workflow may reasonably send a chat-style messages array,
    a single field, or a bare string. Being strict here would turn an
    integration detail into a production outage.
    """
    if isinstance(payload, str):
        return payload
    if isinstance(payload, dict):
        for key in ("conversation", "input", "text", "prompt", "query"):
            if isinstance(payload.get(key), str) and payload[key].strip():
                return payload[key]
        messages = payload.get("messages")
        if isinstance(messages, list):
            parts = []
            for m in messages:
                if isinstance(m, dict) and m.get("content"):
                    role = m.get("role", "user")
                    parts.append(f"{role.capitalize()}: {m['content']}")
            if parts:
                return "\n\n".join(parts)
    raise ValueError(
        "Could not find a conversation in the request. Send {\"conversation\": "
        "\"...\"} or a chat-style {\"messages\": [...]} payload."
    )
