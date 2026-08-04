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

Deliberately its own module and its own toggle (not folded into
rag_pipeline_v2.py or turned on by default) — kept as a separately
benchmarkable arm so it's directly comparable against plain v2, isolating
this ONE variable the same way custom_rag_v2_checked isolates the execution
checker. Only the RETRIEVAL query is rewritten; the actual conversation the
model sees (and the transcript logged) still uses the user's own original
wording — the rewrite exists purely to improve what gets retrieved, not to
change what's being asked.
"""
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

Return ONLY the rewritten query text — no preamble, no quotes, no JSON, no \
explanation.
"""


async def rewrite_query(
    client: httpx.AsyncClient,
    endpoint_url: str,
    headers: dict,
    original_text: str,
) -> str:
    """
    Returns the rewritten query text, or the ORIGINAL text unchanged on any
    failure (bad response, empty content, request error) — fails open,
    matching relevance_filter.py/execution_checker.py: a broken rewrite must
    never make retrieval worse than not rewriting at all.
    """
    try:
        resp = await client.post(
            endpoint_url,
            headers=headers,
            json={
                "messages": [
                    {"role": "system", "content": _REWRITE_SYSTEM},
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
