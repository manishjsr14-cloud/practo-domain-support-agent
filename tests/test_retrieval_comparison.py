"""Comparison evidence must preserve its held-out labels and raw baseline."""

from contextlib import redirect_stdout
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from rag.chunking import ROOT, STRATEGIES
from rag.index import RAGIndex
from rag.queries import CALIBRATION_IN_SCOPE, CALIBRATION_OUT_OF_SCOPE, DEMO_QUERIES, TOP_K
from scripts.compare_retrieval import (
    FREEZE_PATH, METHODS, REPORT_PATH, evaluate_group, load_heldout,
    raw_cosine_retrieve, run_comparison, sha256, verify_frozen_inputs,
)


class HeldoutTests(unittest.TestCase):
    def test_exact_frozen_questions_and_labels_are_isolated(self):
        expected = [
            ("ho01", "My refill request has been submitted; does that mean I am guaranteed the same medicine?",
             ["05_prescription_refills"]),
            ("ho02", "The clinic cancelled my appointment with hardly any notice. What refund can I expect, and when?",
             ["02_cancellation_rescheduling"]),
            ("ho03", "Can a child use the video service from home instead of visiting the clinic?",
             ["07_telemedicine"]),
            ("ho04", "Can I arrange a home appointment and use my discounted return visit for its travel charge?",
             ["12_home_visits", "10_follow_up_discount"]),
        ]
        heldout = load_heldout()
        self.assertEqual([(row["id"], row["query"], row["relevant_parents"]) for row in heldout], expected)
        production = {row["query"] for row in DEMO_QUERIES} | set(CALIBRATION_IN_SCOPE + CALIBRATION_OUT_OF_SCOPE)
        self.assertFalse({row["query"] for row in heldout} & production)
        frozen = json.loads(FREEZE_PATH.read_text(encoding="utf-8"))
        self.assertEqual(frozen["heldout_queries"], heldout)
        verify_frozen_inputs(frozen)

    def test_frozen_input_changes_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = root / "query.json"
            path.write_text("original", encoding="utf-8")
            frozen = {"files": {"query.json": sha256(path)}}
            verify_frozen_inputs(frozen, root)
            path.write_text("changed", encoding="utf-8")
            with self.assertRaisesRegex(RuntimeError, "query.json"):
                verify_frozen_inputs(frozen, root)


class ComparisonTests(unittest.TestCase):
    def test_group_aggregation_deduplicates_parents_before_macro_averages(self):
        def hits(parents):
            return [{"chunk_id": f"{parent}:{position}", "parent_id": parent,
                     "distance": 0.25, "similarity": 0.75}
                    for position, parent in enumerate(parents)]

        class Index:
            def retrieve(self, query, **kwargs):
                return hits(["a", "a", "b"] if query == "first" else ["c", "d", "d"])

        queries = [{"id": "one", "query": "first", "relevant_parents": ["a", "b"]},
                   {"id": "two", "query": "second", "relevant_parents": ["c"]}]
        with patch("scripts.compare_retrieval.raw_cosine_retrieve", return_value=hits(["a", "a", "x"])):
            group = evaluate_group(Index(), queries)
        for strategy in STRATEGIES:
            first = group["queries"][0]["strategies"][strategy]["custom"]
            self.assertEqual(first["precision_denominator"], 2)
            self.assertEqual(first["recall_denominator"], 2)
            self.assertEqual(first["precision"], 1)
            self.assertEqual(first["recall"], 1)
            average = group["averages"][strategy]["custom"]
            self.assertEqual(average["precision"], (1 + 1 / 2) / 2)
            self.assertEqual(average["recall"], 1)

    def test_plain_baseline_equals_direct_chroma_and_preserves_anchor(self):
        index = RAGIndex()
        query = DEMO_QUERIES[0]["query"]
        for strategy in STRATEGIES:
            hits = raw_cosine_retrieve(index, query, strategy)
            direct = index.get_collection(strategy).query(
                query_embeddings=index._embed([query]), n_results=TOP_K,
                include=["metadatas", "distances"],
            )
            anchor = index.retrieve(query, strategy=strategy, top_k=1)[0]
            self.assertEqual([hit["chunk_id"] for hit in hits], direct["ids"][0])
            self.assertEqual([hit["distance"] for hit in hits], direct["distances"][0])
            self.assertEqual(hits[0]["chunk_id"], anchor["chunk_id"])
            self.assertEqual(hits[0]["similarity"], anchor["similarity"])
            self.assertEqual(len(hits), TOP_K)
            for hit in hits:
                self.assertEqual(hit["similarity"], 1 - hit["distance"])

    def test_saved_report_schema_and_arithmetic_do_not_require_target_scores(self):
        report = json.loads(REPORT_PATH.read_text(encoding="utf-8"))
        self.assertEqual(set(report["groups"]), {"demo", "heldout"})
        self.assertEqual(report["top_k_chunks"], TOP_K)
        self.assertEqual(report["groups"]["demo"]["query_count"], len(DEMO_QUERIES))
        self.assertEqual(report["groups"]["heldout"]["query_count"], len(load_heldout()))
        self.assertEqual(report["protected_artifacts_before"], report["protected_artifacts_after"])
        self.assertTrue(report["protected_artifacts_unchanged"])
        for group in report["groups"].values():
            self.assertEqual(group["query_count"], len(group["queries"]))
            for strategy in STRATEGIES:
                for method in METHODS:
                    for row in group["queries"]:
                        metrics = row["strategies"][strategy][method]
                        self.assertEqual(metrics["precision_denominator"], len(set(metrics["retrieved_parents"])))
                        self.assertEqual(metrics["recall_denominator"], len(set(row["relevant_parents"])))
                        numerator = len(set(metrics["retrieved_parents"]) & set(row["relevant_parents"]))
                        self.assertEqual(metrics["numerator"], numerator)
                        self.assertEqual(metrics["precision"], numerator / metrics["precision_denominator"])
                        self.assertEqual(metrics["recall"], numerator / metrics["recall_denominator"])
                        self.assertEqual(metrics["retrieved_chunk_ids"], [hit["chunk_id"] for hit in metrics["hits"]])
                        self.assertEqual(len(metrics["hits"]), TOP_K)
                        for hit in metrics["hits"]:
                            self.assertEqual(hit["similarity"], 1 - hit["distance"])
                    for metric in ("precision", "recall"):
                        values = [row["strategies"][strategy][method][metric] for row in group["queries"]]
                        self.assertAlmostEqual(group["averages"][strategy][method][metric], sum(values) / len(values))
                for row in group["queries"]:
                    methods = row["strategies"][strategy]
                    self.assertEqual(methods["raw_cosine"]["hits"][0], methods["custom"]["hits"][0])
        self.assertNotIn("recommended_strategy", report)

    def test_nonpersisting_comparison_returns_and_prints_without_writing_reports(self):
        recorded = json.loads(REPORT_PATH.read_text(encoding="utf-8"))
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "unused" / "comparison.json"
            printed = io.StringIO()
            with patch("scripts.compare_retrieval.evaluate_group", side_effect=[
                    recorded["groups"]["demo"], recorded["groups"]["heldout"]]), redirect_stdout(printed):
                report = run_comparison(RAGIndex(), output_path=output, persist=False)
            self.assertFalse(output.parent.exists())
        self.assertEqual(report["groups"], recorded["groups"])
        self.assertIn("Diagnostic retrieval comparison", printed.getvalue())
        self.assertTrue(report["protected_artifacts_unchanged"])

    def test_output_cannot_overwrite_calibration_before_measuring(self):
        with patch("scripts.compare_retrieval.evaluate_group") as evaluate:
            with self.assertRaisesRegex(ValueError, "protected artifacts"):
                run_comparison(RAGIndex(), output_path=ROOT / "rag" / "calibration.json")
            evaluate.assert_not_called()


if __name__ == "__main__":
    unittest.main()
