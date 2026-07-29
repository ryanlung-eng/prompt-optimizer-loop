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


def _split_sections(markdown_text: str) -> List[dict]:
    """
    Raw split on every top-level `## ` heading (never `### ` — those are
    sub-points WITHIN a topic, e.g. "Interval Examples" under "Schedule
    Trigger", and splitting there would fragment a single node's docs across
    chunks). Returns [] if the text has no `## ` headers at all, so callers
    can fall back to treating the whole text as one chunk.
    """
    matches = list(_SECTION_RE.finditer(markdown_text))
    sections = []
    for i, m in enumerate(matches):
        start = m.start()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(markdown_text)
        raw_title = m.group(1).strip()
        section_text = markdown_text[start:end].rstrip()

        # Numbered sections look like "1. Workflow Structure" — pull the number
        # out for a stable sort order; unnumbered sections (e.g. "Webhook
        # Trigger", "Non-negotiables (cross-cutting)") keep title as-is.
        num_match = re.match(r"(\d+)\.\s*(.+)$", raw_title)
        if num_match:
            title = num_match.group(2).strip()
        else:
            title = raw_title
        sections.append({"title": title, "text": section_text})
    return sections


def chunk_markdown_by_section(markdown_text: str) -> List[DocChunk]:
    """
    Splits on every top-level `## ` heading. Content before the first `## `
    (the doc's title/intro) is dropped — it's boilerplate, not retrievable detail.
    """
    chunks: List[DocChunk] = []
    for i, sec in enumerate(_split_sections(markdown_text)):
        chunks.append(DocChunk(
            id=f"{i:02d}-{_slugify(sec['title'])}",
            section_number=i,
            title=sec["title"],
            text=sec["text"],
        ))
    return chunks


def load_and_chunk(doc_path: str) -> List[DocChunk]:
    text = Path(doc_path).read_text()
    return chunk_markdown_by_section(text)


def chunk_directory_by_file(dir_path: str, id_prefix: str = "") -> List[DocChunk]:
    """
    Structure-aware, one chunk per topic: for each `.md` file, split on its
    own internal `## ` headers if it has any (each header already scopes one
    node/topic — e.g. "Webhook Trigger", "IF Node" — so this keeps a chunk to
    exactly one topic's embedding instead of diluting it across everything
    in the file); files with no `## ` headers fall back to one chunk for the
    whole file, unchanged from before.

    This matters more than it sounds: several of these files cover 15-20+
    distinct node types each (n8nNodeCatalog-utility.md alone: Manual
    Trigger, Schedule Trigger, Webhook Trigger, IF, Switch, Merge, Code, ...).
    Embedding the whole file as one vector means a query about one specific
    node competes for ranking against an embedding that's an average over
    everything else in the file too — diluted enough that the single most
    relevant file can still miss the top-K for the exact node it actually
    documents. Splitting on the file's own headers first is a strict
    precision improvement with no loss of coverage.

    id_prefix disambiguates IDs when combining chunks from multiple
    directories into one table (e.g. "docs"/"examples") — without it, two
    calls each restart numbering at 000 and their `id` columns collide.
    """
    prefix = f"{id_prefix}-" if id_prefix else ""
    chunks: List[DocChunk] = []
    counter = 0
    for path in sorted(Path(dir_path).glob("*.md")):
        text = path.read_text().strip()
        file_label = path.stem.replace("_", " ").replace("-", " ")
        sections = _split_sections(text)

        if not sections:
            # No internal ## headers — whole file is already one topic.
            chunks.append(DocChunk(
                id=f"{prefix}{counter:03d}-{_slugify(path.stem)}",
                section_number=counter,
                title=file_label,
                text=text,
            ))
            counter += 1
            continue

        for sec in sections:
            # Prefix the chunk title with the source file for context (a
            # bare "Webhook Trigger" title loses which doc family it's from
            # when several files could plausibly have a section with that
            # name) — the file stem alone, not the full section header line.
            title = f"{file_label}: {sec['title']}"
            chunks.append(DocChunk(
                id=f"{prefix}{counter:03d}-{_slugify(title)}",
                section_number=counter,
                title=title,
                text=sec["text"],
            ))
            counter += 1

    return chunks


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
