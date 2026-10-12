"""Threshold calibration and strictly extractive offline generation checks."""

import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from rag.generate import FALLBACK, GroundedGenerator, extract_answer, load_threshold
from rag.index import RAGIndex
from rag.queries import DEMO_OUT_OF_SCOPE, DEMO_QUERIES
from scripts.calibrate_threshold import choose_threshold


class CalibrationTests(unittest.TestCase):
    def test_midpoint_separates_observed_clusters_and_overlap_fails(self):
        positives, negatives = [0.8, 0.75, 0.85], [0.2, 0.3]
        threshold = choose_threshold(positives, negatives)
        self.assertTrue(max(negatives) < threshold < min(positives))
        self.assertEqual(threshold, (max(negatives) + min(positives)) / 2)
        with self.assertRaisesRegex(ValueError, "overlap"):
            choose_threshold([0.2, 0.5, 0.6], [0.3, 0.4])

    def test_stale_calibration_fails_before_generation(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "calibration.json"
            path.write_text(json.dumps({"fingerprint": "obsolete"}), encoding="utf-8")
            with self.assertRaisesRegex(RuntimeError, "Stale calibration"):
                load_threshold(RAGIndex(), "sentence", path)


class ExtractionTests(unittest.TestCase):
    def test_fixed_chunk_starting_at_sentence_boundary_is_preserved(self):
        from rag.chunking import fixed_overlap_chunks, load_documents
        body = load_documents()["07_telemedicine"]
        chunk = fixed_overlap_chunks("07_telemedicine", body)[1]
        self.assertTrue(body[:chunk.start].endswith(". "))
        hit = {"parent_id": chunk.parent_id, "text": chunk.text,
               "chunk_id": chunk.id, "similarity": 1.0,
               "metadata": {"strategy": chunk.strategy, "start": chunk.start,
                            "end": chunk.end, "starts_sentence": True}}
        answer, sources = extract_answer([hit], threshold=0.0)
        self.assertEqual(answer, chunk.text)
        self.assertEqual(sources, [chunk.id])

    def test_partial_sentences_are_excluded_and_overlaps_are_not_repeated(self):
        text = "Complete first sentence. Complete second sentence."
        common = {"parent_id": "p", "similarity": 1.0}
        hits = [dict(common, text=text[:35], chunk_id="a", metadata={
                    "strategy": "fixed_overlap", "start": 0, "end": 35,
                    "starts_sentence": True}),
                dict(common, text=text[20:], chunk_id="b", metadata={
                    "strategy": "fixed_overlap", "start": 20, "end": len(text),
                    "starts_sentence": False})]
        answer, sources = extract_answer(hits, threshold=0.0)
        self.assertEqual(answer, text)


class GenerationIntegrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.index = RAGIndex()
        cls.index.build()

    def test_both_calibrated_strategies_accept_demo_queries_and_refuse_unrelated_query(self):
        for strategy in ("fixed_overlap", "sentence"):
            generator = GroundedGenerator(self.index, strategy=strategy)
            for item in DEMO_QUERIES:
                response = generator.answer(item["query"])
                self.assertFalse(response["fallback"], item["query"])
                self.assertTrue(response["source_chunks"])
                # Every answer sentence must be covered by retrieved source spans.
                from rag.chunking import SENTENCE_SPLIT_PATTERN
                import re
                parent_id = response["retrieved_context"][0]["parent_id"]
                body = self.index.documents[parent_id]
                for sentence in re.split(SENTENCE_SPLIT_PATTERN, response["answer"]):
                    self.assertIn(sentence, body)
                    start = body.index(sentence)
                    for position in range(start, start + len(sentence)):
                        self.assertTrue(any(hit["parent_id"] == parent_id and
                                            hit["metadata"]["start"] <= position < hit["metadata"]["end"]
                                            for hit in response["retrieved_context"]), sentence)
            fallback = generator.answer(DEMO_OUT_OF_SCOPE)
            self.assertTrue(fallback["fallback"])
            self.assertEqual(fallback["answer"], FALLBACK)
            self.assertEqual(fallback["source_chunks"], [])

    def test_real_llm_mode_is_explicitly_rejected(self):
        with patch.dict(os.environ, {"MOCK_LLM": "false"}):
            with self.assertRaisesRegex(ValueError, "MOCK_LLM"):
                GroundedGenerator(self.index)


if __name__ == "__main__":
    unittest.main()
