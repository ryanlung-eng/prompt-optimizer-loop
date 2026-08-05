"""
Pre-retrieval query rewriting for the custom-RAG pipeline.

The KB is written in n8n's own technical vocabulary (node names, parameter
names, resource/operation terms, architectural pattern names like "self-loop
guard" or "approval gate"), which often diverges from how a user phrases an
automation request in plain language (e.g. "make a bot that doesn't reply to
itself" vs. the KB's "anti-loop guard" / "compare bot identity, not content").
Retrieval embeds the raw user request as-is today — this module inserts a
cheap rewrite step before that embedding happens, translating the request
into retrieval-optimized phrasing that's more likely to land near the right
KB vocabulary in embedding space.

GROUNDED in the corpus, not written blind. The caller runs a cheap
first-pass retrieval on the user's own wording and passes the resulting
document titles/sources in as candidates, so the rewrite is steered by
vocabulary the KB demonstrably contains (classic pseudo-relevance feedback).
This exists because the blind version was caught, in a live benchmark run,
inventing "OpenAI" into 4 of 17 queries — a provider this platform has no
credential for — which pulled the wrong docs and correlated with a real
blocker regression. A prompt-level denylist of absent vendors (also below)
treats that symptom; showing the rewriter the actual corpus removes the
opportunity, since it can see what vocabulary is really there. Degrades
cleanly: with no candidates passed, it behaves exactly like the blind
version.

Deliberately its own module and its own toggle (not folded into
rag_pipeline_v2.py or turned on by default) — kept as a separately
benchmarkable arm so it's directly comparable against plain v2, isolating
this ONE variable the same way custom_rag_v2_checked isolates the execution
checker. Only the RETRIEVAL query is rewritten; the actual conversation the
model sees (and the transcript logged) still uses the user's own original
wording — the rewrite exists purely to improve what gets retrieved, not to
change what's being asked.
"""
from typing import List, Optional

import httpx

_REWRITE_SYSTEM = """\
You are rewriting a user's natural-language automation request into a short, \
retrieval-optimized search query for a knowledge base of n8n node and \
workflow-pattern documentation. The KB is written in n8n's own technical \
vocabulary — node names, parameter names, resource/operation terms, and \
named architectural patterns (e.g. "anti-loop guard", "approval gate", \
"Structured Output Parser") — which often differs from how a user phrases \
the same request in plain language.

Rewrite the request into a query that surfaces the RIGHT technical \
vocabulary: name the node types, n8n-specific terms, and pattern names the \
request most likely implies. Do not invent specifics you're not confident \
about, and do not answer the request or describe how to build it — only \
produce a short (1-3 sentence) search query capturing what the request \
needs documentation for.

CRITICAL — never introduce the name of a vendor, provider, or product that \
does not appear in the original request. This platform has exactly these \
integrations available: Slack, Gmail, Google Sheets/Docs/Drive/Slides/ \
Calendar, Jira, and Databricks (the ONLY LLM provider — every AI step uses \
a Databricks chat model). Writing "OpenAI", "GPT", "Anthropic", "Postgres", \
"Notion", or any other absent provider into the query retrieves the wrong \
documentation and steers the build toward an integration that does not \
exist here. For an AI/LLM step, say "AI Agent node" or "LLM node" — never \
a provider name the user did not use.

When a list of CANDIDATE KB DOCUMENTS is provided below, treat it as \
ground truth about what vocabulary this knowledge base actually uses: those \
titles and filenames are real documents retrieved for this request. Prefer \
their terminology, and name the specific topics among them that this \
request needs. Do not use a technical term that contradicts them, and do \
not assume a document exists for a technology absent from that list."""

_CANDIDATES_BLOCK = """\

CANDIDATE KB DOCUMENTS — a first-pass retrieval on the user's own wording \
returned these. They show the vocabulary and topic space actually available:
{candidates}

Rewrite the request into a query that pulls the RIGHT subset of this \
material (and any closely-related topics these titles imply), using their \
vocabulary."""

_RETURN_INSTRUCTION = """\

Return ONLY the rewritten query text — no preamble, no quotes, no JSON, no \
explanation."""

_MAX_CANDIDATES = 12


async def rewrite_query(
    client: httpx.AsyncClient,
    endpoint_url: str,
    headers: dict,
    original_text: str,
    candidate_docs: Optional[List[str]] = None,
) -> str:
    """
    Returns the rewritten query text, or the ORIGINAL text unchanged on any
    failure (bad response, empty content, request error) — fails open,
    matching relevance_filter.py/execution_checker.py: a broken rewrite must
    never make retrieval worse than not rewriting at all.

    candidate_docs: preformatted "Title (source)" strings from a cheap
    first-pass retrieval on the RAW user text. Passing them grounds the
    rewrite in vocabulary the corpus actually contains (see module
    docstring); omitting them falls back to blind rewriting.
    """
    system = _REWRITE_SYSTEM
    if candidate_docs:
        listed = "\n".join(f"- {c}" for c in candidate_docs[:_MAX_CANDIDATES])
        system += _CANDIDATES_BLOCK.format(candidates=listed)
    system += _RETURN_INSTRUCTION
    try:
        resp = await client.post(
            endpoint_url,
            headers=headers,
            json={
                "messages": [
                    {"role": "system", "content": system},
                    {"role": "user", "content": original_text},
                ],
                "max_tokens": 300,
                "temperature": 0.0,
            },
            timeout=60,
        )
        resp.raise_for_status()
        body = resp.json()
        choices = body.get("choices") or []
        content = (choices[0].get("message") or {}).get("content") if choices else None
        if not content or not content.strip():
            return original_text
        return content.strip()
    except Exception:
        return original_text
