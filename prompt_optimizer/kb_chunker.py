"""
Splits instructions-digest.md (or instructions.md) into one chunk per top-level
`## ` section — these already correspond to one integration/topic each (Gmail,
Slack, Credentials, ...), so this is a structural split, not a fixed-size window.
Chunking on the doc's own section boundaries is the fix for the "right document,
window too small" bug: a chunk here is always a complete section's worth of
detail, never a mid-node-table cut.

Pure Python, no Spark/Databricks imports — importable and testable locally,
and imported by rag_setup.py (a Databricks notebook) to build the Delta table.
"""
import re
from dataclasses import dataclass
from pathlib import Path
from typing import List

_SECTION_RE = re.compile(r"^## (.+)$", re.MULTILINE)


@dataclass
class DocChunk:
    id: str            # stable slug, e.g. "06-gmail"
    section_number: int  # 0 for unnumbered sections (e.g. "Non-negotiables")
    title: str          # e.g. "Gmail"
    text: str           # full section body, including the "## Title" line

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "section_number": self.section_number,
            "title": self.title,
            "text": self.text,
        }


def _slugify(title: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", title.lower()).strip("-")
    return slug


def chunk_markdown_by_section(markdown_text: str) -> List[DocChunk]:
    """
    Splits on every top-level `## ` heading. Content before the first `## `
    (the doc's title/intro) is dropped — it's boilerplate, not retrievable detail.
    """
    matches = list(_SECTION_RE.finditer(markdown_text))
    chunks: List[DocChunk] = []

    for i, m in enumerate(matches):
        start = m.start()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(markdown_text)
        raw_title = m.group(1).strip()
        section_text = markdown_text[start:end].rstrip()

        # Numbered sections look like "1. Workflow Structure" — pull the number
        # out for a stable sort order; unnumbered sections (e.g. "Non-negotiables
        # (cross-cutting)") get section_number 0 and sort last.
        num_match = re.match(r"(\d+)\.\s*(.+)$", raw_title)
        if num_match:
            section_number = int(num_match.group(1))
            title = num_match.group(2).strip()
        else:
            section_number = 0
            title = raw_title

        chunks.append(DocChunk(
            id=f"{section_number:02d}-{_slugify(title)}",
            section_number=section_number,
            title=title,
            text=section_text,
        ))

    return chunks


def load_and_chunk(doc_path: str) -> List[DocChunk]:
    text = Path(doc_path).read_text()
    return chunk_markdown_by_section(text)


if __name__ == "__main__":
    import json
    import sys

    path = sys.argv[1] if len(sys.argv) > 1 else "instructions-digest.md"
    chunks = load_and_chunk(path)
    print(f"{len(chunks)} chunks from {path}")
    for c in chunks:
        print(f"  [{c.id}] {c.title} — {len(c.text)} chars")
    if "--dump" in sys.argv:
        json.dump([c.to_dict() for c in chunks], sys.stdout, indent=2)
