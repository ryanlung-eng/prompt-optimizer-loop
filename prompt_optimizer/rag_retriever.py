"""
Retrieval step for the custom RAG pipeline — queries the AI Search index built
by rag_setup.py and assembles a token-budgeted context block to inject into the
Workflow Builder prompt, replacing the "paste the whole KB into every prompt"
approach.

Plain Python module (not a notebook) — imported directly by evaluator.py /
loop.py, same as any other pipeline component in this package.
"""
from dataclasses import dataclass, field
from typing import List

from tenacity import retry, stop_after_attempt, wait_random_exponential


@dataclass
class RAGConfig:
    # Whether the caller (loop.py/evaluator.py) should retrieve context at all
    # — false until rag_setup.py has been run once and the index is live.
    enabled: bool = False
    endpoint_name: str = "n8n-kb-endpoint"
    index_name: str = "dev.platform.automation_builder_kb_chunks_index"
    # The frozen, always-injected core (cross-cutting rules) — deliberately
    # OUTSIDE docs/examples so chunk_directory_by_file's glob never picks it
    # up as a retrievable chunk. If it lived inside docs/, it would only get
    # pulled in when it happens to rank in a query's top-K, defeating the
    # point of having a guaranteed-present core.
    instructions_path: str = "/Volumes/dev/platform/automation-builder/instructions.md"
    # Raised from 5: hard-scenario benchmarking showed 5 was too thin across
    # the merged docs+examples corpus — a query would miss the one chunk
    # covering the specific node it needed (webhook, Jira update, Trello
    # Trigger), leaving the model to guess and hallucinate params instead of
    # having the real schema in front of it. Set to 10 to mirror Ibotta's AI
    # & Automation team's own HR Bot (unfiltered hybrid retrieval, top-K=10
    # on a comparable corpus) — dial back down if 10 turns out to be overkill
    # for a corpus this size, but starting from their proven value rather
    # than guessing.
    top_k: int = 10
    # Character budget, not a token budget — see retrieve_context() docstring
    # for why. ~25000 chars is a conservative ~6250-token ceiling at a ~4
    # chars/token English-text estimate — raised alongside top_k=10 so 10
    # retrieved chunks aren't immediately truncated by a budget sized for the
    # old top_k=5/8.
    max_context_chars: int = 25000
    query_type: str = "hybrid"   # ANN + keyword (RRF) — Databricks' recommended default
    use_reranker: bool = True
    # Generation model for the custom-RAG arm specifically — deliberately
    # separate from DatabricksConfig.generation_endpoint/fast_generation_endpoint
    # (which serve prompt-optimization and simulated-user replies respectively)
    # so this can be swapped per-run (e.g. to an Opus endpoint) via config.yaml
    # alone, without touching code. Was previously silently defaulting to
    # WorkflowEvaluator._generation_url (Haiku, fast_generation_endpoint) —
    # every custom-RAG benchmark run before this field existed was generating
    # on Haiku, not Sonnet.
    generation_endpoint: str = "databricks-claude-sonnet-4-6"
    # Retrieval-time diversity cap: at most this many of the top_k results may
    # come from the same source document. Without this, one large file that
    # scores well overall for a broad query can fill most/all of the top-K
    # slots, starving coverage of other relevant docs — e.g. a big node-
    # catalog file crowding out the one chunk from a different file that
    # actually covers the specific integration a request needs.
    max_chunks_per_source: int = 4
    # How many raw candidates to fetch before applying the per-source cap —
    # needs to comfortably exceed top_k so there's still enough of a pool left
    # to fill top_k slots after over-represented sources get capped.
    over_fetch_multiplier: int = 3
    # Source documents whose retrieved chunks the v2 relevance filter is never
    # allowed to drop (see rag_pipeline_v2.retrieve_and_filter). These cover
    # cross-cutting platform infrastructure — which credentials exist, how to
    # wire the Databricks chat model, how AI Agent sub-node slots connect —
    # that is required for EVERY workflow but is never topically "about" the
    # request, so a filter told to keep only chunks that directly answer the
    # question will discard them.
    #
    # Not hypothetical: benchmarking showed the filtered arm falling back to
    # hallucinated OpenAI nodes (lmChatOpenAi/gpt-4o — no such credential on
    # this platform) on the exact scenarios where the unfiltered arm correctly
    # used lmChatDatabricks, because the "Databricks Chat Model Sub-Node"
    # chunk doesn't look relevant to e.g. "classify support emails".
    #
    # Tuple, not list — dataclass defaults must be immutable.
    protected_sources: tuple = (
        "n8nNodeCatalog-credentials",
        "n8nNodeCatalog-databricks",
        "n8nNodeCatalog-ai_agent",
    )
    # Sources injected UNCONDITIONALLY, before the query is even considered.
    #
    # protected_sources alone was not enough, and the benchmark proved it: it
    # can only re-add a chunk that retrieval already surfaced and the filter
    # then dropped. For a query like "classify support emails", the Databricks
    # chat-model chunk never enters the top-K at all, so there is nothing to
    # protect — and the filtered arm produced 8 workflows wired to OpenAI
    # (a provider with no credential here) against 1 for the unfiltered arm.
    #
    # Deliberately NARROWER than protected_sources: only the two docs that
    # answer "which credential and which node type do I use for an LLM here",
    # since this cost is paid on every single request.
    always_inject_sources: tuple = (
        "n8nNodeCatalog-credentials",
        "n8nNodeCatalog-databricks",
    )
    # Hard cap on always-injected chunks so the fixed cost stays bounded —
    # those two sources are ~7 chunks total and we only need the wiring ones.
    max_always_inject_chunks: int = 4


@dataclass
class RetrievedChunk:
    id: str
    title: str
    text: str
    source: str


def _get_index(config: RAGConfig):
    from databricks.ai_search.client import AISearchClient
    client = AISearchClient()
    return client.get_index(index_name=config.index_name)


# similarity_search re-embeds query_text at REQUEST TIME via the Delta Sync
# Index's embedding_source_column config, hitting the databricks-gte-large-en
# Model Serving endpoint on every single call — not just at index-build time.
# A benchmark run issues this concurrently across every scenario AND every
# repair turn (more arms/turns = more concurrent load on the SAME shared
# endpoint), which surfaces exactly the transient failure Model Serving's
# own docs describe under load: "INTERNAL_ERROR: Could not route request" —
# the router couldn't find a healthy backend for this one request, not a
# malformed call (that would be a 4xx-style INVALID_PARAMETER_VALUE instead).
# Retried with the same jittered backoff as evaluator.py's LLM calls, since
# this was previously completely unretried and a single transient hiccup
# anywhere in a whole batch crashed the entire asyncio.gather() call instead
# of just that one item.
@retry(stop=stop_after_attempt(6), wait=wait_random_exponential(multiplier=1, min=4, max=60))
def retrieve_chunks(query_text: str, config: RAGConfig = RAGConfig()) -> List[RetrievedChunk]:
    """
    Top-K retrieval, ranked best-first by similarity_search, with a per-source
    diversity cap applied afterward: fetches top_k * over_fetch_multiplier raw
    candidates, then greedily keeps rank order while skipping any candidate
    whose source has already hit max_chunks_per_source, stopping once top_k
    slots are filled (or candidates run out — a query where one source
    legitimately dominates every relevant result just returns fewer than
    top_k rather than backfilling with irrelevant filler).
    """
    index = _get_index(config)

    kwargs = dict(
        query_text=query_text,
        columns=["id", "title", "text", "source"],
        num_results=config.top_k * config.over_fetch_multiplier,
        query_type=config.query_type,
    )
    if config.use_reranker:
        from databricks.ai_search.reranker import DatabricksReranker
        kwargs["reranker"] = DatabricksReranker(columns_to_rerank=["text"])

    results = index.similarity_search(**kwargs)
    rows = results["result"]["data_array"]
    ranked = [RetrievedChunk(id=r[0], title=r[1], text=r[2], source=r[3]) for r in rows]

    selected: List[RetrievedChunk] = []
    per_source_count: dict = {}
    for c in ranked:
        if per_source_count.get(c.source, 0) >= config.max_chunks_per_source:
            continue
        selected.append(c)
        per_source_count[c.source] = per_source_count.get(c.source, 0) + 1
        if len(selected) >= config.top_k:
            break
    return selected


# Fixed query used only to LOCATE the always-inject chunks. Retrieval is still
# the mechanism (no new SDK surface, no direct Delta read from this module),
# but the query is constant and the results are filtered to
# always_inject_sources — so what comes back does not depend on the user's
# request, which is the entire point.
_INFRA_QUERY = (
    "Databricks chat model sub-node credential type, credential ID table, "
    "AI Agent language model connection wiring"
)


def retrieve_always_injected(config: RAGConfig = RAGConfig()) -> List[RetrievedChunk]:
    """
    Chunks that must be present for EVERY request regardless of what was
    asked — which credentials exist and how to wire the Databricks chat
    model. Returns [] on any failure: missing grounding degrades output,
    but a retrieval error here must never fail the whole request.
    """
    sources = tuple(getattr(config, "always_inject_sources", ()) or ())
    if not sources:
        return []
    try:
        index = _get_index(config)
        results = index.similarity_search(
            query_text=_INFRA_QUERY,
            columns=["id", "title", "text", "source"],
            num_results=config.top_k * config.over_fetch_multiplier,
            query_type=config.query_type,
        )
        rows = results["result"]["data_array"]
    except Exception as e:
        print(f"  Warning: always-inject retrieval failed ({e}) — continuing without it.")
        return []
    hits = [
        RetrievedChunk(id=r[0], title=r[1], text=r[2], source=r[3])
        for r in rows if r[3] in sources
    ]
    return hits[: config.max_always_inject_chunks]


def format_chunks(chunks: List[RetrievedChunk], max_context_chars: int) -> str:
    """
    Formats an already-selected, already-ordered list of chunks into a single
    context block, trimmed to max_context_chars (chunks dropped from the end
    first — callers are expected to pass chunks best-first). Split out of
    retrieve_context() so a caller that wants to filter/reorder chunks before
    formatting (e.g. rag_pipeline_v2.py's relevance filter) can reuse the same
    budget-trimming logic instead of reimplementing it.
    """
    assembled: List[str] = []
    total_chars = 0
    for c in chunks:
        block = f"### {c.title}\n{c.text}"
        if total_chars + len(block) > max_context_chars and assembled:
            # Keep at least one chunk even if it alone exceeds the budget — a
            # single retrieved section is still better than an empty context.
            break
        assembled.append(block)
        total_chars += len(block)

    return "\n\n".join(assembled)


def retrieve_context(query_text: str, config: RAGConfig = RAGConfig()) -> str:
    """
    Returns a single formatted context block for the retrieved KB sections,
    trimmed to config.max_context_chars (lowest-relevance chunks dropped first
    — similarity_search already ranks results best-first).

    Character-based budget, not a real tokenizer: the actual generation call
    goes through a Databricks-hosted Claude serving endpoint (Mosaic AI), not
    the Anthropic API directly, so there's no local `count_tokens` call
    available here to measure exactly.
    """
    chunks = retrieve_chunks(query_text, config)
    return format_chunks(chunks, config.max_context_chars)
