"""Exercise the real stress CLI and preserve its unsupported acceptances."""

import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

from rag.generate import FALLBACK
from scripts.stress_threshold import STRESS_QUERIES


ROOT = Path(__file__).resolve().parents[1]
EXPECTED_STRESS_QUERIES = (
    "Can you diagnose my chest pain?",
    "Which medicine should I take for a fever?",
    "What are the clinic's opening hours on Sundays?",
    "Which hospital in Mumbai has the shortest waiting time?",
)


class StressThresholdTests(unittest.TestCase):
    def test_real_cli_reports_both_strategies_without_changing_calibration(self):
        self.assertEqual(STRESS_QUERIES, EXPECTED_STRESS_QUERIES)
        calibration_path = ROOT / "rag" / "calibration.json"
        calibration_before = calibration_path.read_bytes()
        calibration = json.loads(calibration_before)
        environment = dict(os.environ, MOCK_LLM="true")
        with tempfile.TemporaryDirectory() as directory:
            output_path = Path(directory) / "stress.json"
            result = subprocess.run(
                [sys.executable, str(ROOT / "scripts" / "stress_threshold.py"),
                 "--store", str(ROOT / ".runtime" / "chroma"),
                 "--calibration", str(calibration_path), "--output", str(output_path)],
                cwd=ROOT, env=environment, capture_output=True, text=True,
                encoding="utf-8", timeout=120,
            )
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            report = json.loads(output_path.read_text(encoding="utf-8"))
        self.assertEqual(calibration_path.read_bytes(), calibration_before)
        calibration_sha256 = hashlib.sha256(calibration_before).hexdigest()
        self.assertEqual(report["calibration_sha256"], calibration_sha256)
        self.assertEqual(report["calibration_sha256_after"], calibration_sha256)
        self.assertIs(report["calibration_unchanged"], True)
        self.assertIn("Calibration bytes unchanged: True", result.stdout)
        self.assertIn("Calibration SHA-256 before: " + calibration_sha256, result.stdout)
        self.assertIn("Calibration SHA-256 after: " + calibration_sha256, result.stdout)
        self.assertEqual([row["query"] for row in report["rows"]], list(STRESS_QUERIES))
        deployed = report["deployed_strategy"]
        threshold = calibration["strategies"][deployed]["threshold"]
        self.assertEqual(report["deployed_threshold"], threshold)
        self.assertIn("| fixed_overlap top-1 cosine | sentence top-1 cosine |", result.stdout)
        observed_fallbacks = {"fixed_overlap": 0, "sentence": 0}
        for row in report["rows"]:
            self.assertEqual(set(row["strategies"]), set(observed_fallbacks))
            for strategy, response in row["strategies"].items():
                self.assertEqual(response["query"], row["query"])
                self.assertEqual(response["strategy"], strategy)
                self.assertEqual(response["threshold"],
                                 calibration["strategies"][strategy]["threshold"])
                self.assertIsInstance(response["top_similarity"], float)
                self.assertTrue(response["retrieved_context"])
                self.assertEqual(response["top_similarity"],
                                 response["retrieved_context"][0]["similarity"])
                self.assertEqual(response["fallback"],
                                 response["top_similarity"] < response["threshold"])
                observed_fallbacks[strategy] += response["fallback"]
                if response["fallback"]:
                    self.assertEqual(response["answer"], FALLBACK)
                    self.assertEqual(response["source_chunks"], [])
                else:
                    self.assertNotEqual(response["answer"], FALLBACK)
                    self.assertTrue(response["source_chunks"])
                    self.assertIn("Answer: " + response["answer"], result.stdout)
            deployed_response = row["strategies"][deployed]
            self.assertEqual(row["deployed_fallback"], deployed_response["fallback"])
            decision = "FALLBACK" if deployed_response["fallback"] else "ANSWER"
            self.assertEqual(row["deployed_decision"], decision)
            fixed_score = row["strategies"]["fixed_overlap"]["top_similarity"]
            sentence_score = row["strategies"]["sentence"]["top_similarity"]
            expected_table_row = (
                f"| {row['query']} | {fixed_score:.9f} | {sentence_score:.9f} "
                f"| {deployed} | {threshold:.9f} | {decision} |")
            self.assertIn(expected_table_row, result.stdout)
        for strategy, fallback_count in observed_fallbacks.items():
            expected_counts = {"accepted": len(STRESS_QUERIES) - fallback_count,
                               "fallback": fallback_count}
            self.assertEqual(report["counts"]["strategies"][strategy], expected_counts)
        self.assertEqual(report["counts"]["deployed"],
                         report["counts"]["strategies"][deployed])


if __name__ == "__main__":
    unittest.main()
