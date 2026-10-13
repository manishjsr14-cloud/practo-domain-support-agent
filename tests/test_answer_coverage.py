"""Regression checks for paraphrases and questions spanning clinic policies."""

import unittest
from unittest.mock import patch

from rag.generate import GroundedGenerator
from rag.index import RAGIndex
from rag.queries import DEMO_QUERIES, CALIBRATION_IN_SCOPE, CALIBRATION_OUT_OF_SCOPE
from rag.retrieval import question_focus, question_components


class AnswerCoverageTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.index = RAGIndex()
        cls.index.build()

    def _response(self, query_id, strategy):
        query = next(row for row in DEMO_QUERIES if row["id"] == query_id)
        response = GroundedGenerator(self.index, strategy).answer(query["query"])
        self.assertFalse(response["fallback"])
        retrieved_parents = {hit["parent_id"] for hit in response["retrieved_context"]}
        self.assertTrue(set(query["relevant_parents"]) <= retrieved_parents)
        source_map = {hit["chunk_id"]: hit["parent_id"] for hit in response["retrieved_context"]}
        cited_parents = {source_map[source] for source in response["source_chunks"]}
        self.assertTrue(set(query["relevant_parents"]) <= cited_parents)
        self.assertEqual(len(response["retrieved_context"]), 3)
        return response

    def test_follow_up_paraphrase_retrieves_and_cites_discount_policy(self):
        for strategy in ("fixed_overlap", "sentence"):
            with self.subTest(strategy=strategy):
                response = self._response("q07", strategy)
                self.assertIn("20 percent", response["answer"])
                self.assertIn("7 calendar days", response["answer"])
                self.assertEqual({section["parent_id"] for section in response["context_sections"]},
                                 {"10_follow_up_discount"})

    def test_cardiologist_home_visit_includes_fees_and_separate_travel_quote(self):
        for strategy in ("fixed_overlap", "sentence"):
            with self.subTest(strategy=strategy):
                response = self._response("q08", strategy)
                self.assertIn("Cardiology consultations cost INR 800-1500", response["answer"])
                self.assertIn("any travel charge are quoted before the visit is confirmed", response["answer"])

    def test_video_second_opinion_includes_process_and_telemedicine_limits(self):
        for strategy in ("fixed_overlap", "sentence"):
            with self.subTest(strategy=strategy):
                response = self._response("q09", strategy)
                self.assertIn("Request a second opinion through reception", response["answer"])
                self.assertIn("routine follow-ups", response["answer"])
                self.assertIn("clinician confirms", response["answer"])

    def test_additional_paraphrases_cite_the_required_policies(self):
        cases = [
            ("Does a repeat appointment cost less than the original visit?", {"10_follow_up_discount"}),
            ("How much is a cardiology appointment at home, including travel?",
             {"03_consultation_fees", "12_home_visits"}),
            ("May I obtain another specialist's opinion via a video appointment?",
             {"11_second_opinion", "07_telemedicine"}),
        ]
        for strategy in ("fixed_overlap", "sentence"):
            for query, expected in cases:
                with self.subTest(strategy=strategy, query=query):
                    response = GroundedGenerator(self.index, strategy).answer(query)
                    self.assertFalse(response["fallback"])
                    cited = {section["parent_id"] for section in response["context_sections"]}
                    self.assertTrue(expected <= cited, (query, cited))

    def test_original_cosine_anchor_is_independent_of_context_ranking(self):
        for strategy in ("fixed_overlap", "sentence"):
            for query in CALIBRATION_IN_SCOPE + CALIBRATION_OUT_OF_SCOPE:
                with self.subTest(strategy=strategy, query=query):
                    anchor = self.index.retrieve(query, strategy, top_k=1)[0]
                    context = self.index.retrieve(query, strategy, top_k=3)
                    self.assertEqual(context[0]["chunk_id"], anchor["chunk_id"])
                    self.assertEqual(context[0]["similarity"], anchor["similarity"])
                    self.assertEqual(context[0]["distance"], anchor["distance"])

    def test_grammar_helpers_do_not_need_clinic_topics_or_query_ids(self):
        query = "My laptop needs repair; can I book a home visit?"
        self.assertEqual(question_focus(query), "can I book a home visit?")
        self.assertEqual(question_components(query), ["book", "home visit"])
        qualified = "How much is a repair at home, including travel?"
        self.assertEqual(question_focus(qualified), qualified)

    def test_coverage_cannot_override_whole_query_fallback(self):
        class LowSimilarityIndex:
            def retrieve(self, *args, **kwargs):
                return [{"similarity": 0.1, "text": "Some context."}]

            def _embed(self, *args):
                raise AssertionError("Coverage must not run on a rejected question")

        with patch("rag.generate.load_threshold", return_value=0.5):
            response = GroundedGenerator(LowSimilarityIndex(), "sentence").answer("An unsupported question")
        self.assertTrue(response["fallback"])
        self.assertEqual(response["context_sections"], [])
        self.assertIsNone(response["coverage"])


if __name__ == "__main__":
    unittest.main()
