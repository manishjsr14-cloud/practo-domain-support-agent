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
    def test_fixed_demo_queries_and_labels_are_preserved(self):
        expected = [
            ("q01", "How do I book an appointment and know that my slot is confirmed?", ["01_appointment_booking"]),
            ("q02", "What is the cancellation and rescheduling window?", ["02_cancellation_rescheduling"]),
            ("q03", "What are the consultation fees for each specialty?", ["03_consultation_fees"]),
            ("q04", "How long do routine and specialized lab test reports take?", ["06_lab_turnaround"]),
            ("q05", "When does a follow-up visit qualify for a discount?", ["10_follow_up_discount"]),
            ("q06", "How do I submit an insurance reimbursement claim?", ["04_insurance_claims"]),
            ("q07", "My doctor asked me to come back, do I have to pay the full price again?", ["10_follow_up_discount"]),
            ("q08", "What will I pay for a home visit by a cardiologist?", ["12_home_visits", "03_consultation_fees"]),
            ("q09", "Can I get a second opinion over a video call?", ["11_second_opinion", "07_telemedicine"]),
        ]
        actual = [(row["id"], row["query"], row["relevant_parents"]) for row in DEMO_QUERIES]
        self.assertEqual(actual, expected)
        self.assertEqual(TOP_K, 3)

    def test_multi_parent_recall_counts_each_relevant_parent_once(self):
        metrics = document_metrics(["fees", "fees", "home"], ["home", "fees"])
        self.assertEqual(metrics["numerator"], 2)
        self.assertEqual(metrics["precision_denominator"], 2)
        self.assertEqual(metrics["recall_denominator"], 2)
        self.assertEqual(metrics["precision"], 1)
        self.assertEqual(metrics["recall"], 1)
        missing = document_metrics(["fees", "fees", "other"], ["home", "fees"])
        self.assertEqual(missing["recall"], 1 / 2)

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
        self.assertEqual(report["single_parent_query_count"], 7)
        self.assertEqual(report["multi_parent_query_count"], 2)
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
