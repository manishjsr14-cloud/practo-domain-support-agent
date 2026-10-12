"""Deterministic extractive MOCK_LLM; cosine similarity alone gates the answer."""

import json
import os
from pathlib import Path
import re

from rag.index import RAGIndex
from rag.queries import TOP_K

ROOT = Path(__file__).resolve().parents[1]
FALLBACK = "I don't know based on the available clinic-policy context."


def deployed_strategy() -> str:
    path = ROOT / "rag" / "deployment.json"
    return json.loads(path.read_text(encoding="utf-8"))["strategy"] if path.exists() else "sentence"


def load_threshold(index: RAGIndex, strategy: str,
                   path: Path = ROOT / "rag" / "calibration.json") -> float:
    if not path.exists():
        raise RuntimeError("Missing calibration; run python scripts/calibrate_threshold.py")
    data = json.loads(path.read_text(encoding="utf-8"))
    if data["fingerprint"] != index.corpus_fingerprint:
        raise RuntimeError("Stale calibration; rebuild the index and recalibrate")
    return float(data["strategies"][strategy]["threshold"])


def extract_answer(hits: list[dict], threshold: float) -> tuple[str, list[str]]:
    """Quote retrieved text from the most similar parent, without adding facts.

    Merge overlapping spans to avoid repeated prose. Sentence chunks keep all
    their sentences; fixed chunks omit potentially cut leading/trailing text.
    No external document text or keyword-based groundedness check is used.
    """
    parent = hits[0]["parent_id"]
    selected = [hit for hit in hits if hit["parent_id"] == parent
                and hit["similarity"] >= threshold]
    selected.sort(key=lambda hit: hit["metadata"]["start"])
    spans = []
    for hit in selected:
        start, end = hit["metadata"]["start"], hit["metadata"]["end"]
        if spans and start <= spans[-1][1]:
            old_start, old_end, old_text, starts_sentence = spans[-1]
            if end > old_end:
                spans[-1] = (old_start, end, old_text + hit["text"][old_end - start:], starts_sentence)
        else:
            spans.append((start, end, hit["text"], hit["metadata"].get("starts_sentence", False)))
    passages = []
    for start, end, text, starts_sentence in spans:
        if hits[0]["metadata"]["strategy"] == "fixed_overlap":
            if start > 0 and not starts_sentence:
                boundary = re.search(r"[.!?]\s+", text)
                text = text[boundary.end():] if boundary else ""
            trailing = max(text.rfind("."), text.rfind("!"), text.rfind("?"))
            text = text[:trailing + 1] if trailing >= 0 else ""
        if text.strip():
            passages.append(text.strip())
    # Even an unusually small fixed chunk remains a direct context excerpt.
    answer = " ".join(passages) if passages else hits[0]["text"]
    return answer, [hit["chunk_id"] for hit in selected]


class GroundedGenerator:
    def __init__(self, index: RAGIndex, strategy: str | None = None,
                 calibration_path: Path = ROOT / "rag" / "calibration.json"):
        if os.environ.get("MOCK_LLM", "true").lower() not in {"true", "1"}:
            raise ValueError("Part 1 supports only deterministic MOCK_LLM=true")
        self.index = index
        self.strategy = strategy or deployed_strategy()
        self.calibration_path = calibration_path

    def answer(self, query: str, top_k: int = TOP_K) -> dict:
        # Recheck calibration against the corpus for every answer.
        threshold = load_threshold(self.index, self.strategy, self.calibration_path)
        hits = self.index.retrieve(query, strategy=self.strategy, top_k=top_k)
        accepted = bool(hits) and hits[0]["similarity"] >= threshold
        answer, sources = extract_answer(hits, threshold) if accepted else (FALLBACK, [])
        return {"query": query, "answer": answer, "fallback": not accepted,
                "strategy": self.strategy, "threshold": threshold,
                "top_similarity": hits[0]["similarity"] if hits else None,
                "source_chunks": sources, "retrieved_context": hits}
