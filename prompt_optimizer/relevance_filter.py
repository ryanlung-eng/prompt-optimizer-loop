"""
Post-retrieval relevance filter for the custom-RAG pipeline.

Modeled directly on Ibotta's own internal HR Bot (a mature, production
Databricks RAG bot — source read directly, not just described secondhand):
its own team credits this step, not retrieval tuning, as doing most of their
accuracy work. It's a cheap LLM call that looks at the raw top-K retrieved
chunks and picks out only the ones that actually help build/fix THIS
specific request, before the (expensive, reasoning-heavy) generation call
ever sees them — a chunk that's topically related but doesn't supply a
concrete detail the request needs is dropped instead of adding noise.

Deliberately its own module, not folded into rag_retriever.py or
evaluator.py — kept separate so the ORIGINAL custom-RAG arm (evaluator.py's
run_batch_custom_rag) stays completely unchanged, and rag_pipeline_v2.py can
layer this on top as a distinct, separately-benchmarkable arm.
"""
import json
from dataclasses import dataclass
from typing import List

import httpx

from .rag_retriever import RetrievedChunk

_FILTER_SYSTEM = """\
You are filtering retrieved reference documentation before it's handed to an \
n8n workflow-building assistant. You will see the user's request and a \
numbered list of retrieved documentation chunks. Pick ONLY the chunks that \
directly help build or fix THIS specific request — the exact node types, \
parameter names, or patterns it actually needs.

Be strict: drop chunks that are topically related but don't move the \
assistant toward correctly building this specific workflow (e.g. a chunk \
about a different node's credentials, or general background that doesn't \
supply a concrete detail this request needs). When genuinely unsure whether \
a chunk is needed, keep it — dropping a chunk that was actually necessary is \
worse than keeping one extra chunk.

Return ONLY valid JSON: {"useful_indices": [<int>, ...]} — the 1-based \
indices (matching the numbers shown) of the chunks to keep, in any order. \
Return an empty list only if truly none of the chunks help.
"""


@dataclass
class FilterResult:
    kept: List[RetrievedChunk]
    dropped_count: int


def _format_chunks_for_filter(chunks: List[RetrievedChunk]) -> str:
    # Preview only, capped — this call just needs enough of each chunk to
    # judge relevance, not the full text (that's what generation gets).
    return "\n\n".join(f"[{i}] {c.title}\n{c.text[:1500]}" for i, c in enumerate(chunks, start=1))


def _resolve_indices(raw_indices, n: int) -> List[int]:
    """
    Keeps only in-range, 1-based integer indices, silently dropping anything
    else. Mirrors a real bug HR Bot's own relevance filter hit and fixed: a
    hallucinated 0 or negative index used as a bare Python list index
    (items[i-1]) silently wraps via negative indexing instead of erroring —
    so invalid indices must be filtered out explicitly rather than trusted.
    """
    return [i for i in raw_indices if isinstance(i, int) and 1 <= i <= n]


async def filter_relevant_chunks(
    client: httpx.AsyncClient,
    endpoint_url: str,
    headers: dict,
    query_text: str,
    chunks: List[RetrievedChunk],
) -> FilterResult:
    """
    Calls a fast model (pass the Haiku/fast_generation_endpoint URL —
    matching HR Bot's own classify/filter model choice, a stronger model is
    reserved for generation only) to pick which retrieved chunks are
    actually useful for this specific request. Returns the original
    RetrievedChunk objects for the kept indices, in their original
    (already rank-ordered) sequence — never reordered by the filter.

    Fails open on any error (bad JSON, empty response, request failure, or
    every returned index being invalid): returns every chunk unfiltered. A
    relevance-filter bug should degrade to today's unfiltered behavior, not
    silently starve the generation call of all context.
    """
    if not chunks:
        return FilterResult(kept=[], dropped_count=0)

    user = f"Request:\n{query_text}\n\nRetrieved chunks:\n{_format_chunks_for_filter(chunks)}"
    try:
        resp = await client.post(
            endpoint_url,
            headers=headers,
            json={
                "messages": [
                    {"role": "system", "content": _FILTER_SYSTEM},
                    {"role": "user", "content": user},
                ],
                "max_tokens": 500,
                "temperature": 0.0,
            },
            timeout=60,
        )
        resp.raise_for_status()
        body = resp.json()
        choices = body.get("choices") or []
        content = (choices[0].get("message") or {}).get("content") if choices else None
        if not content:
            raise ValueError(f"Empty filter response: {json.dumps(body)[:300]}")
        start, end = content.find("{"), content.rfind("}")
        if start == -1 or end == -1 or end < start:
            raise ValueError(f"No JSON object in filter response: {content[:300]}")
        parsed = json.loads(content[start:end + 1])
        raw_indices = parsed.get("useful_indices", [])
    except Exception:
        return FilterResult(kept=list(chunks), dropped_count=0)

    valid_indices = _resolve_indices(raw_indices, len(chunks))
    if not valid_indices:
        # Every index was invalid/hallucinated, or the model genuinely
        # returned an empty list — fail open rather than hand generation
        # zero context either way.
        return FilterResult(kept=list(chunks), dropped_count=0)

    kept = [chunks[i - 1] for i in sorted(valid_indices)]
    return FilterResult(kept=kept, dropped_count=len(chunks) - len(kept))
