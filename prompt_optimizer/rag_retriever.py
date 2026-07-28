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


@dataclass
class RAGConfig:
    # Whether the caller (loop.py/evaluator.py) should retrieve context at all
    # — false until rag_setup.py has been run once and the index is live.
    enabled: bool = False
    endpoint_name: str = "n8n-kb-endpoint"
    index_name: str = "main.n8n_kb.doc_chunks_index"
    top_k: int = 5
    # Character budget, not a token budget — see retrieve_context() docstring
    # for why. ~12000 chars is a conservative ~3000-token ceiling at a ~4
    # chars/token English-text estimate, leaving headroom rather than cutting
    # it close.
    max_context_chars: int = 12000
    query_type: str = "hybrid"   # ANN + keyword (RRF) — Databricks' recommended default
    use_reranker: bool = True


@dataclass
class RetrievedChunk:
    id: str
    title: str
    text: str


def _get_index(config: RAGConfig):
    from databricks.ai_search.client import AISearchClient
    client = AISearchClient()
    return client.get_index(index_name=config.index_name)


def retrieve_chunks(query_text: str, config: RAGConfig = RAGConfig()) -> List[RetrievedChunk]:
    """Raw top-K retrieval, ranked best-first by similarity_search — no budget trimming."""
    index = _get_index(config)

    kwargs = dict(
        query_text=query_text,
        columns=["id", "title", "text"],
        num_results=config.top_k,
        query_type=config.query_type,
    )
    if config.use_reranker:
        from databricks.ai_search.reranker import DatabricksReranker
        kwargs["reranker"] = DatabricksReranker(columns_to_rerank=["text"])

    results = index.similarity_search(**kwargs)
    rows = results["result"]["data_array"]
    return [RetrievedChunk(id=r[0], title=r[1], text=r[2]) for r in rows]


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

    assembled: List[str] = []
    total_chars = 0
    for c in chunks:
        block = f"### {c.title}\n{c.text}"
        if total_chars + len(block) > config.max_context_chars and assembled:
            # Keep at least one chunk even if it alone exceeds the budget — a
            # single retrieved section is still better than an empty context.
            break
        assembled.append(block)
        total_chars += len(block)

    return "\n\n".join(assembled)
