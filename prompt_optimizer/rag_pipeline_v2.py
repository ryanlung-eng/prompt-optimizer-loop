"""
"v2" custom-RAG context assembly — layers two changes on top of
rag_retriever.py's base retrieval, both modeled on Ibotta's own internal HR
Bot (a mature, production Databricks RAG bot whose source was read directly
to validate these choices, not just described secondhand):

  1. A post-retrieval relevance filter (relevance_filter.py) — HR Bot's own
     team credits this step, not retrieval tuning, as doing most of their
     accuracy work: a cheap Haiku call that drops retrieved chunks that are
     topically related but don't actually help this specific request, before
     the (expensive, reasoning-heavy) generation call ever sees them.
  2. Explicit grounding language appended to the context block itself,
     adapted from HR Bot's synthesizer prompt ("ground every claim in the
     chunks, do not add facts/steps that aren't in them, do not guess").

Kept in its own module rather than merged into evaluator.py's
run_batch_custom_rag / rag_retriever.py's retrieve_context, specifically so
the ORIGINAL custom-RAG arm stays byte-for-byte unchanged — benchmark.py
runs both as separate arms (custom_rag vs custom_rag_v2) so the two
approaches are directly, apples-to-apples comparable, not just replaced.
"""
from typing import List, Tuple

import httpx

from .rag_retriever import (
    RAGConfig, RetrievedChunk, format_chunks, retrieve_chunks, retrieve_always_injected,
)
from .relevance_filter import filter_relevant_chunks

GROUNDING_NOTE = (
    "Ground every parameter name, node type, and behavior claim in the "
    "documentation above — do not invent or guess at details this retrieved "
    "context doesn't cover. If something this request needs genuinely isn't "
    "covered here, say so rather than guessing a plausible-looking value."
)


async def retrieve_and_filter(
    client: httpx.AsyncClient,
    filter_endpoint_url: str,
    filter_headers: dict,
    query_text: str,
    rag_config: RAGConfig,
) -> Tuple[List[RetrievedChunk], List[RetrievedChunk]]:
    """
    Retrieves rag_config.top_k chunks exactly like the base pipeline, then
    filters them down via an LLM call before the caller formats/budget-trims
    them — instead of handing every retrieved chunk straight to generation.
    Returns (retrieved, kept) so the caller can report how many were dropped.

    Chunks from rag_config.protected_sources are re-added after filtering if
    the filter dropped them (see that field for the full rationale): they
    cover platform infrastructure needed by EVERY workflow — which
    credentials exist, how to wire the Databricks chat model — but are never
    topically "about" the request, so a filter selecting for direct relevance
    reliably discards them and the model falls back to hallucinated OpenAI
    nodes. Restored chunks keep their original retrieval rank rather than
    being appended, so the best-first ordering format_chunks() relies on for
    budget-trimming still holds.
    """
    retrieved = retrieve_chunks(query_text, rag_config)
    if not retrieved:
        return [], []
    result = await filter_relevant_chunks(client, filter_endpoint_url, filter_headers, query_text, retrieved)

    protected = tuple(getattr(rag_config, "protected_sources", ()) or ())
    if protected:
        kept_ids = {c.id for c in result.kept}
        kept = [c for c in retrieved if c.id in kept_ids or c.source in protected]
    else:
        kept = list(result.kept)

    # Always-inject runs on a FIXED query, so it does not depend on what the
    # user asked and cannot be starved by a request that makes infrastructure
    # look irrelevant — the failure protected_sources could not cover, since
    # that only rescues chunks retrieval already returned. Prepended (not
    # appended) because format_chunks trims from the END when the character
    # budget is hit, and this grounding must survive that trim.
    infra = retrieve_always_injected(rag_config)
    if infra:
        have = {c.id for c in kept}
        kept = [c for c in infra if c.id not in have] + kept

    return retrieved, kept


def build_retrieved_block(kept: List[RetrievedChunk], n_retrieved: int, rag_config: RAGConfig) -> str:
    """
    Formats the FILTERED chunk list (not the raw retrieval) into the context
    block injected into the system prompt, with the grounding note appended.
    """
    context = format_chunks(kept, rag_config.max_context_chars)
    return (
        f"Retrieved reference documentation ({len(kept)} of {n_retrieved} top-"
        f"{rag_config.top_k} candidates selected as relevant to this specific "
        f"request — use this as the authoritative source for exact parameter "
        f"names and node behavior beyond what's already above):\n\n{context}"
        f"\n\n{GROUNDING_NOTE}"
    )
