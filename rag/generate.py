"""Deterministic extractive MOCK_LLM; cosine similarity alone gates the answer."""

import json
import os
from pathlib import Path
import re

from rag.index import RAGIndex
from rag.queries import TOP_K
from rag.retrieval import COVERAGE_POLICY, question_components, question_focus

ROOT = Path(__file__).resolve().parents[1]
FALLBACK = "I don't know based on the available clinic-policy context."


def deployed_strategy(index: RAGIndex, path: Path = ROOT / "rag" / "deployment.json") -> str:
    if not path.exists():
        return "sentence"
    data = json.loads(path.read_text(encoding="utf-8"))
    if data["fingerprint"] != index.corpus_fingerprint:
        raise RuntimeError("Stale deployment choice; rerun python -m rag.evaluate")
    if data.get("retrieval_fingerprint") != index.retrieval_fingerprint:
        raise RuntimeError("Stale retrieval policy; rerun python -m rag.evaluate")
    return data["strategy"]


def load_threshold(index: RAGIndex, strategy: str,
                   path: Path = ROOT / "rag" / "calibration.json") -> float:
    if not path.exists():
        raise RuntimeError("Missing calibration; run python scripts/calibrate_threshold.py")
    data = json.loads(path.read_text(encoding="utf-8"))
    if data["fingerprint"] != index.corpus_fingerprint:
        raise RuntimeError("Stale calibration; rebuild the index and recalibrate")
    return float(data["strategies"][strategy]["threshold"])


def select_evidence(query: str, hits: list[dict], index: RAGIndex):
    """Choose coverage after the original whole-query gate has accepted.

    Parent semantics select the primary policy. Other parents may contribute
    when they best match a grammatical content phrase in the question. Phrase
    scores are coverage heuristics, not a newly calibrated refusal signal.
    """
    import numpy as np

    parent_scores = {}
    for hit in hits:
        parent_scores[hit["parent_id"]] = max(
            parent_scores.get(hit["parent_id"], float("-inf")),
            hit.get("parent_relevance", hit["similarity"]))
    primary = max(parent_scores, key=parent_scores.get)
    selected_parents = {primary}
    components = question_components(query)
    evidence = []
    if components:
        scores = np.asarray(index._embed(components)) @ np.asarray(
            index._embed([hit["text"] for hit in hits])).T
        for component, similarities in zip(components, scores):
            best = hits[int(np.argmax(similarities))]
            similarity = float(np.max(similarities))
            primary_similarity = max(float(similarities[i]) for i, hit in enumerate(hits)
                                     if hit["parent_id"] == primary)
            # Secondary excerpts need both a semantic win and a textual anchor.
            # A phrase can match on one content word while its other words are
            # paraphrases; the primary policy still uses full-question semantics.
            lexical_anchor = any(re.search(r"\b" + re.escape(word) + r"\b",
                                           best["text"], re.IGNORECASE)
                                 for word in component.split())
            selected = best["parent_id"] == primary or (
                similarity > primary_similarity and lexical_anchor)
            if selected:
                selected_parents.add(best["parent_id"])
            evidence.append({"component": component, "similarity": similarity,
                             "parent_id": best["parent_id"], "chunk_id": best["chunk_id"],
                             "primary_similarity": primary_similarity,
                             "lexical_anchor": lexical_anchor, "selected": selected})
    ordered = [primary] + sorted(selected_parents - {primary}, key=lambda p: -parent_scores[p])
    selected_hits = [hit for parent in ordered for hit in hits if hit["parent_id"] == parent]
    return selected_hits, {"policy": COVERAGE_POLICY, "question_focus": question_focus(query),
                           "primary_parent": primary, "selected_parents": ordered,
                           "components": evidence}


def _extract_parent(selected: list[dict]) -> tuple[str, list[str]]:
    """Merge retrieved spans, omitting cut sentence fragments and repetition."""
    selected = sorted(selected, key=lambda hit: hit["metadata"]["start"])
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
    sources = []
    for start, end, text, starts_sentence in spans:
        visible_start, visible_end = start, end
        if selected[0]["metadata"]["strategy"] == "fixed_overlap":
            if start > 0 and not starts_sentence:
                boundary = re.search(r"[.!?]\s+", text)
                visible_start += boundary.end() if boundary else len(text)
                text = text[boundary.end():] if boundary else ""
            trailing = max(text.rfind("."), text.rfind("!"), text.rfind("?"))
            text = text[:trailing + 1] if trailing >= 0 else ""
            visible_end = visible_start + len(text)
        if text.strip():
            passages.append(text.strip())
            sources.extend(hit["chunk_id"] for hit in selected
                           if hit["metadata"]["start"] < visible_end
                           and hit["metadata"]["end"] > visible_start)
    # Even an unusually small fixed chunk remains a direct context excerpt.
    if not passages:
        return selected[0]["text"], [selected[0]["chunk_id"]]
    return " ".join(passages), list(dict.fromkeys(sources))


def extract_sections(hits: list[dict]) -> list[dict]:
    """Present each selected policy as its own verbatim retrieved excerpt."""
    sections = []
    for parent in dict.fromkeys(hit["parent_id"] for hit in hits):
        text, sources = _extract_parent([hit for hit in hits if hit["parent_id"] == parent])
        sections.append({"parent_id": parent, "text": text, "source_chunks": sources})
    return sections


def extract_answer(hits: list[dict]) -> tuple[str, list[str]]:
    sections = extract_sections(hits)
    return "\n\n".join(section["text"] for section in sections), [
        source for section in sections for source in section["source_chunks"]]


class GroundedGenerator:
    def __init__(self, index: RAGIndex, strategy: str | None = None,
                 calibration_path: Path = ROOT / "rag" / "calibration.json"):
        if os.environ.get("MOCK_LLM", "true").lower() not in {"true", "1"}:
            raise ValueError("Part 1 supports only deterministic MOCK_LLM=true")
        self.index = index
        self.strategy = strategy or deployed_strategy(index)
        self.calibration_path = calibration_path

    def answer(self, query: str, top_k: int = TOP_K) -> dict:
        # Recheck calibration against the corpus for every answer.
        threshold = load_threshold(self.index, self.strategy, self.calibration_path)
        hits = self.index.retrieve(query, strategy=self.strategy, top_k=top_k)
        accepted = bool(hits) and hits[0]["similarity"] >= threshold
        sections, coverage = [], None
        answer, sources = FALLBACK, []
        if accepted:
            selected, coverage = select_evidence(query, hits, self.index)
            sections = extract_sections(selected)
            answer = "\n\n".join(section["text"] for section in sections)
            sources = [source for section in sections for source in section["source_chunks"]]
        return {"query": query, "answer": answer, "fallback": not accepted,
                "strategy": self.strategy, "threshold": threshold,
                "top_similarity": hits[0]["similarity"] if hits else None,
                "source_chunks": sources, "retrieved_context": hits,
                "context_sections": sections, "coverage": coverage}
