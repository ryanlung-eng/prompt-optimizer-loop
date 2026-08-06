"""
Normalising the `content` field of a chat-completions response.

Every endpoint in this project is called through the chat-completions
envelope, where `choices[0].message.content` was assumed to be a plain
string. It is not always: newer Claude endpoints can return Anthropic-style
CONTENT BLOCKS — a list of {"type": "text", "text": "..."} dicts — inside
that same envelope. Callers then did `content.strip()` and got

    'list' object has no attribute 'strip'

which surfaced mid-benchmark as a whole scenario failing. Every one of the
eight call sites had the same latent assumption, including the judge, where
it would have taken out scoring rather than one generation.

Blocks are JOINED rather than first-match, since a reply can legitimately be
split across several text blocks and taking only the first would silently
truncate a workflow. Non-text blocks (reasoning/thinking) are skipped: they
are not part of the answer, and concatenating them would corrupt JSON output
with prose the model never intended to emit.
"""
from typing import Any, Optional

# Block types whose "text" field is part of the actual answer. A block with no
# declared type is treated as text — some gateways omit it — but a block that
# names itself something else (thinking, reasoning, tool_use) is not the reply.
_TEXT_BLOCK_TYPES = (None, "text", "output_text", "input_text")


def content_to_text(content: Any) -> Optional[str]:
    """Return the assistant's text, whether the endpoint sent a string or
    a list of content blocks. None when there is no usable text at all, so
    callers keep their existing "unrecognised/empty response" handling."""
    if content is None:
        return None
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for block in content:
            if isinstance(block, str):
                parts.append(block)
            elif isinstance(block, dict):
                if block.get("type") in _TEXT_BLOCK_TYPES and block.get("text"):
                    parts.append(block["text"])
        return "".join(parts) or None
    return None
