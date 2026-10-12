"""Two transparent chunkers over the same normalized knowledge-base bodies."""

from dataclasses import dataclass
from pathlib import Path
import re
from typing import Mapping


ROOT = Path(__file__).resolve().parents[1]
FIXED_CHUNK_SIZE = 320
FIXED_CHUNK_OVERLAP = 80
SENTENCES_PER_CHUNK = 2
SENTENCE_SPLIT_PATTERN = r"(?<=[.!?])\s+"
STRATEGIES = ("fixed_overlap", "sentence")


@dataclass(frozen=True)
class Chunk:
    """Offsets refer to the normalized parent body, with an exclusive end."""

    id: str
    parent_id: str
    text: str
    strategy: str
    index: int
    start: int
    end: int


def normalize_document(text: str) -> str:
    """Remove Markdown headings and collapse whitespace for stable offsets."""
    body_lines = [line for line in text.splitlines()
                  if not line.lstrip().startswith("#")]
    return " ".join(" ".join(body_lines).split())


def load_documents(kb_dir: Path = ROOT / "kb") -> dict[str, str]:
    """Load one parent document per Markdown file in filename order."""
    paths = sorted(Path(kb_dir).glob("*.md"))
    if not paths:
        raise ValueError(f"No Markdown knowledge-base documents found in {kb_dir}")
    documents = {}
    for path in paths:
        body = normalize_document(path.read_text(encoding="utf-8"))
        if not body:
            raise ValueError(f"Knowledge-base document has no body: {path}")
        documents[path.stem] = body
    return documents


def fixed_overlap_chunks(
    parent_id: str,
    body: str,
    size: int = FIXED_CHUNK_SIZE,
    overlap: int = FIXED_CHUNK_OVERLAP,
) -> list[Chunk]:
    """Split at character boundaries, preserving overlapping source offsets."""
    if size <= 0 or not 0 <= overlap < size:
        raise ValueError("Chunk size must be positive and overlap smaller than size")
    chunks = []
    for window_start in range(0, len(body), size - overlap):
        window_end = min(window_start + size, len(body))
        raw = body[window_start:window_end]
        text = raw.strip()
        if text:
            start = window_start + len(raw) - len(raw.lstrip())
            end = window_end - (len(raw) - len(raw.rstrip()))
            index = len(chunks)
            chunks.append(Chunk(
                id=f"{parent_id}:fixed_overlap:{index:03d}",
                parent_id=parent_id, text=text, strategy="fixed_overlap",
                index=index, start=start, end=end,
            ))
        if window_end == len(body):
            break
    return chunks


def sentence_chunks(
    parent_id: str,
    body: str,
    sentences_per_chunk: int = SENTENCES_PER_CHUNK,
) -> list[Chunk]:
    """Group consecutive complete sentences without overlap."""
    if sentences_per_chunk <= 0:
        raise ValueError("Sentences per chunk must be positive")
    sentences = [part for part in re.split(SENTENCE_SPLIT_PATTERN, body) if part]
    spans = []
    cursor = 0
    for sentence in sentences:
        start = body.index(sentence, cursor)
        end = start + len(sentence)
        spans.append((start, end))
        cursor = end
    chunks = []
    for offset in range(0, len(spans), sentences_per_chunk):
        group = spans[offset:offset + sentences_per_chunk]
        start, end = group[0][0], group[-1][1]
        index = len(chunks)
        chunks.append(Chunk(
            id=f"{parent_id}:sentence:{index:03d}",
            parent_id=parent_id, text=body[start:end], strategy="sentence",
            index=index, start=start, end=end,
        ))
    return chunks


def chunk_documents(documents: Mapping[str, str], strategy: str) -> list[Chunk]:
    """Apply a named strategy to every parent using stable iteration order."""
    chunkers = {"fixed_overlap": fixed_overlap_chunks, "sentence": sentence_chunks}
    if strategy not in chunkers:
        raise ValueError(f"Unknown strategy {strategy!r}; choose from {STRATEGIES}")
    return [chunk for parent_id, body in sorted(documents.items())
            for chunk in chunkers[strategy](parent_id, body)]
