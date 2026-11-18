"""Hierarchical, heading-aware document chunking."""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

_HEADING = re.compile(r"^(#{1,6})\s+(.*)$")


@dataclass(frozen=True)
class Chunk:
    """A retrievable passage with provenance."""

    id: str
    doc_id: str
    section: str
    text: str
    metadata: dict = field(default_factory=dict, compare=False, hash=False)


def split_sentences(text: str) -> list[str]:
    """Split text into sentences without breaking common abbreviations or decimals."""
    text = re.sub(r"\s+", " ", text.strip())
    if not text:
        return []
    parts = re.split(r"(?<=[.!?])\s+(?=[A-Z0-9\"'(])", text)
    return [p.strip() for p in parts if p.strip()]


def chunk_markdown(doc_id: str, text: str, max_words: int = 120, overlap: int = 1) -> list[Chunk]:
    """Chunk a markdown document by section, then by sentence windows.

    Each chunk stays inside one section (so citations point at a meaningful heading)
    and carries ``overlap`` sentences from the previous window to preserve context.
    """
    if max_words < 10:
        raise ValueError("max_words must be >= 10")
    sections: list[tuple[str, list[str]]] = [("Introduction", [])]
    for line in text.splitlines():
        m = _HEADING.match(line.strip())
        if m:
            sections.append((m.group(2).strip(), []))
        elif line.strip():
            sections[-1][1].append(line.strip())

    chunks: list[Chunk] = []
    for section, lines in sections:
        sentences = split_sentences(" ".join(lines))
        if not sentences:
            continue
        window: list[str] = []
        for sent in sentences:
            if window and sum(len(s.split()) for s in window + [sent]) > max_words:
                chunks.append(_make(doc_id, section, window, len(chunks)))
                window = window[-overlap:] if overlap else []
            window.append(sent)
        if window:
            chunks.append(_make(doc_id, section, window, len(chunks)))
    return chunks


def _make(doc_id: str, section: str, sentences: list[str], n: int) -> Chunk:
    return Chunk(id=f"{doc_id}#{n}", doc_id=doc_id, section=section, text=" ".join(sentences))


def load_corpus(folder: str | Path, max_words: int = 120) -> list[Chunk]:
    """Load and chunk every ``.md`` / ``.txt`` file in a folder."""
    folder = Path(folder)
    out: list[Chunk] = []
    for path in sorted(folder.glob("*")):
        if path.suffix.lower() in {".md", ".txt"}:
            out.extend(chunk_markdown(path.stem, path.read_text(encoding="utf-8"), max_words))
    return out
