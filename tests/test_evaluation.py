"""Document metrics must deduplicate chunks before choosing deployment."""

from contextlib import redirect_stdout
import io
import json
from pathlib import Path
import tempfile
import unittest

from rag.evaluate import _select_strategy, document_metrics, run_evaluation
from rag.index import RAGIndex
from rag.generate import deployed_strategy
from rag.queries import DEMO_QUERIES, TOP_K


class EvaluationTests(unittest.TestCase):
    def test_deployment_choice_must_match_current_corpus(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "deployment.json"
            path.write_text(json.dumps({"fingerprint": "obsolete", "strategy": "sentence"}), encoding="utf-8")
            with self.assertRaisesRegex(RuntimeError, "Stale deployment"):
                deployed_strategy(RAGIndex(), path)

    def test_duplicate_parents_do_not_inflate_precision_or_recall(self):
        metrics = document_metrics(["a", "a", "b"], ["a", "c"])
        self.assertEqual(metrics["numerator"], 1)
        self.assertEqual(metrics["precision_denominator"], 2)
        self.assertEqual(metrics["recall_denominator"], 2)
        self.assertEqual(metrics["precision"], 1 / 2)
        self.assertEqual(metrics["recall"], 1 / 2)
        self.assertEqual(document_metrics([], ["a"])["precision"], 0)

    def test_selection_prefers_recall_then_precision_then_sentence_tie(self):
        averages = {"fixed_overlap": {"precision": 0.5, "recall": 1.0},
                    "sentence": {"precision": 0.8, "recall": 0.5}}
        self.assertEqual(_select_strategy(averages)[0], "fixed_overlap")
        averages["sentence"]["recall"] = 1.0
        self.assertEqual(_select_strategy(averages)[0], "sentence")
        averages["sentence"]["precision"] = 0.5
        self.assertEqual(_select_strategy(averages)[0], "sentence")

    def test_real_report_uses_exact_shared_queries_and_document_denominators(self):
        index = RAGIndex()
        index.build()
        with redirect_stdout(io.StringIO()):
            report = run_evaluation(index, persist=False)
        self.assertEqual([row["query"] for row in report["queries"]],
                         [row["query"] for row in DEMO_QUERIES])
        self.assertEqual(report["top_k_chunks"], TOP_K)
        for row in report["queries"]:
            for strategy, metrics in row["strategies"].items():
                self.assertEqual(metrics["precision_denominator"], len(set(metrics["retrieved_parents"])))
                self.assertEqual(metrics["recall_denominator"], len(set(row["relevant_parents"])))
                self.assertLessEqual(metrics["precision_denominator"], TOP_K)
        for strategy in ("fixed_overlap", "sentence"):
            for metric in ("precision", "recall"):
                values = [row["strategies"][strategy][metric] for row in report["queries"]]
                self.assertAlmostEqual(report["averages"][strategy][metric], sum(values) / len(values))


if __name__ == "__main__":
    unittest.main()
