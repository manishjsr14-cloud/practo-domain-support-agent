"""Part 1 checks use unittest so dataset/KB checks need no dependencies."""

from collections import Counter
from pathlib import Path
import random
import shutil
import tempfile
import unittest

from dataset import (
    APPOINTMENTS, CATEGORY_WEIGHTS, FEE_RANGES, RECORD_COUNT, STATUS_WEIGHTS,
    generate_appointments, validate_appointments,
)
from scripts.validate_kb import STANDARD_FEES, TOPICS, validate_fee_policy, validate_kb
from rag.chunking import (
    FIXED_CHUNK_OVERLAP, FIXED_CHUNK_SIZE, chunk_documents,
    fixed_overlap_chunks, load_documents, sentence_chunks,
)


class DatasetTests(unittest.TestCase):
    def test_deterministic_and_independent_of_global_rng(self):
        random.seed(123)
        random.random()
        self.assertEqual(APPOINTMENTS, generate_appointments())
        self.assertNotEqual(APPOINTMENTS, generate_appointments(seed=43))

    def test_counts_and_coverage(self):
        validate_appointments(APPOINTMENTS)
        self.assertEqual(len(APPOINTMENTS), RECORD_COUNT)
        categories = Counter(row["category"] for row in APPOINTMENTS)
        statuses = Counter(row["status"] for row in APPOINTMENTS)
        self.assertTrue(all(categories[name] >= 3 for name in CATEGORY_WEIGHTS))
        self.assertTrue(all(statuses[name] >= 1 for name in STATUS_WEIGHTS))
        reserved = APPOINTMENTS[:len(CATEGORY_WEIGHTS)]
        self.assertEqual({row["category"] for row in reserved}, set(CATEGORY_WEIGHTS))
        self.assertEqual({row["status"] for row in reserved}, set(STATUS_WEIGHTS))
        fraction = sum(row["follow_up_required"] for row in APPOINTMENTS) / RECORD_COUNT
        self.assertTrue(0.10 <= fraction <= 0.30)

    def test_schema_identifiers_and_ranges(self):
        fields = {"record_id", "category", "status", "consultation_fee_inr",
                  "days_since_created", "follow_up_required"}
        self.assertEqual(len({row["record_id"] for row in APPOINTMENTS}), RECORD_COUNT)
        for row in APPOINTMENTS:
            self.assertEqual(set(row), fields)
            self.assertRegex(row["record_id"], r"^APT-\d{3}$")
            low, high = FEE_RANGES[row["category"]]
            self.assertIs(type(row["consultation_fee_inr"]), int)
            self.assertTrue(low <= row["consultation_fee_inr"] <= high)
            self.assertEqual(row["consultation_fee_inr"] % 50, 0)
            self.assertIs(type(row["days_since_created"]), int)
            self.assertTrue(0 <= row["days_since_created"] <= 30)
            self.assertIs(type(row["follow_up_required"]), bool)


class KnowledgeBaseTests(unittest.TestCase):
    @staticmethod
    def fee_policy() -> str:
        schedule = ", ".join(f"INR {fee} for {specialty}"
                             for specialty, fee in STANDARD_FEES.items())
        return (
            f"New bookings use these standard consultation fees: {schedule}. "
            "The recorded consultation fee in an existing booking is authoritative "
            "and may differ from the standard consultation fee because of an adjustment."
        )

    def test_every_topic_has_a_document_with_valid_sentences_and_fees(self):
        results = validate_kb()
        self.assertEqual({name for name, count in results}, set(TOPICS))
        self.assertTrue(all(2 <= count <= 5 for name, count in results))

    def test_missing_topics_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(ValueError, "topic files"):
                validate_kb(Path(directory))

    def test_fixed_standard_fees_stay_within_unchanged_dataset_ranges(self):
        self.assertEqual(STANDARD_FEES, {
            "General Medicine": 500, "Cardiology": 1200, "Dermatology": 800,
            "Pediatrics": 600, "Orthopedics": 1000,
        })
        self.assertEqual(validate_fee_policy(self.fee_policy()), STANDARD_FEES)
        for specialty, fee in STANDARD_FEES.items():
            low, high = FEE_RANGES[specialty]
            self.assertTrue(low <= fee <= high)

    def test_in_range_but_incorrect_standard_fee_is_rejected(self):
        policy = self.fee_policy().replace("INR 500 for General Medicine",
                                          "INR 550 for General Medicine")
        with self.assertRaisesRegex(ValueError, "standard fee mismatch.*General Medicine"):
            validate_fee_policy(policy)

    def test_standard_fee_outside_dataset_range_is_rejected(self):
        policy = self.fee_policy().replace("INR 500 for General Medicine",
                                          "INR 650 for General Medicine")
        with self.assertRaisesRegex(ValueError, "outside dataset fee range.*General Medicine"):
            validate_fee_policy(policy)

    def test_existing_recorded_fee_authority_cannot_be_removed_or_negated(self):
        for replacement in ("illustrative", "not authoritative"):
            with self.subTest(replacement=replacement):
                policy = self.fee_policy().replace("is authoritative", "is " + replacement)
                with self.assertRaisesRegex(ValueError, "recorded-fee authority"):
                    validate_fee_policy(policy)

    def test_new_booking_and_existing_booking_distinction_is_required(self):
        for original, replacement in (("New bookings", "Bookings"),
                                      ("existing booking", "booking")):
            with self.subTest(original=original):
                policy = self.fee_policy().replace(original, replacement)
                with self.assertRaises(ValueError):
                    validate_fee_policy(policy)

    def test_topic_title_cannot_replace_privacy_policy_content(self):
        with tempfile.TemporaryDirectory() as directory:
            copied = Path(directory) / "kb"
            shutil.copytree(Path(__file__).resolve().parents[1] / "kb", copied)
            (copied / "09_patient_privacy.md").write_text(
                "# Patient privacy - fictional clinic\n\n"
                "Patients can book appointments. Staff confirm a time.\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "missing required topic content"):
                validate_kb(copied)

    def test_sentence_limits_are_still_enforced(self):
        with tempfile.TemporaryDirectory() as directory:
            copied = Path(directory) / "kb"
            shutil.copytree(Path(__file__).resolve().parents[1] / "kb", copied)
            (copied / "01_appointment_booking.md").write_text(
                "# Appointment booking - fictional clinic\n\n"
                "Book an appointment through staff.\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "sentence count outside required range"):
                validate_kb(copied)


class ChunkingTests(unittest.TestCase):
    def test_fixed_windows_cover_source_with_exact_overlap(self):
        body = "a" * (FIXED_CHUNK_SIZE * 3)
        chunks = fixed_overlap_chunks("sample", body)
        self.assertEqual(chunks[0].start, 0)
        self.assertEqual(chunks[-1].end, len(body))
        for previous, current in zip(chunks, chunks[1:]):
            self.assertEqual(previous.end - current.start, FIXED_CHUNK_OVERLAP)
            self.assertEqual(previous.text[-FIXED_CHUNK_OVERLAP:],
                             current.text[:FIXED_CHUNK_OVERLAP])
        self.assertEqual(fixed_overlap_chunks("empty", ""), [])
        with self.assertRaises(ValueError):
            fixed_overlap_chunks("bad", "text", size=10, overlap=10)

    def test_sentence_chunks_preserve_boundaries_and_short_tail(self):
        body = "First sentence. Second sentence? Third sentence!"
        chunks = sentence_chunks("sample", body)
        self.assertEqual([chunk.text for chunk in chunks],
                         ["First sentence. Second sentence?", "Third sentence!"])
        self.assertEqual(sentence_chunks("empty", ""), [])

    def test_all_chunk_offsets_match_parent_text(self):
        documents = load_documents()
        for strategy in ("fixed_overlap", "sentence"):
            chunks = chunk_documents(documents, strategy)
            self.assertEqual(len({chunk.id for chunk in chunks}), len(chunks))
            self.assertEqual({chunk.parent_id for chunk in chunks}, set(documents))
            for chunk in chunks:
                self.assertEqual(documents[chunk.parent_id][chunk.start:chunk.end], chunk.text)


class IndexTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from rag.index import RAGIndex
        cls.temporary = tempfile.TemporaryDirectory(ignore_cleanup_errors=True)
        cls.directory = Path(cls.temporary.name)
        cls.kb_dir = cls.directory / "kb"
        shutil.copytree(Path(__file__).resolve().parents[1] / "kb", cls.kb_dir)
        cls.index = RAGIndex(store_path=cls.directory / "chroma", kb_dir=cls.kb_dir)
        cls.counts = cls.index.build()

    @classmethod
    def tearDownClass(cls):
        cls.temporary.cleanup()

    def test_separate_cosine_collections_with_parent_metadata_and_stable_upserts(self):
        collections = [self.index.get_collection(name) for name in ("fixed_overlap", "sentence")]
        self.assertNotEqual(collections[0].id, collections[1].id)
        self.assertEqual(self.counts, self.index.build())
        documents = self.index.documents
        for collection in collections:
            self.assertEqual(collection.configuration["hnsw"]["space"], "cosine")
            stored = collection.get(include=["metadatas", "documents"])
            for text, metadata in zip(stored["documents"], stored["metadatas"]):
                self.assertIn(metadata["parent_id"], documents)
                self.assertEqual(text, documents[metadata["parent_id"]][metadata["start"]:metadata["end"]])

    def test_retrieval_uses_cosine_distance_and_both_find_cancellation_policy(self):
        import numpy as np
        query = "What is the cancellation and rescheduling window?"
        query_vector = self.index._embed([query])[0]
        for strategy in ("fixed_overlap", "sentence"):
            hits = self.index.retrieve(query, strategy, top_k=3)
            self.assertEqual(hits[0]["parent_id"], "02_cancellation_rescheduling")
            self.assertEqual(hits[0]["similarity"], 1.0 - hits[0]["distance"])
            vector = self.index.get_collection(strategy).get(
                ids=[hits[0]["chunk_id"]], include=["embeddings"])["embeddings"][0]
            cosine = np.dot(query_vector, vector) / (np.linalg.norm(query_vector) * np.linalg.norm(vector))
            self.assertAlmostEqual(float(cosine), hits[0]["similarity"], places=5)

    def test_changed_corpus_is_rejected_until_rebuilt(self):
        path = self.kb_dir / "01_appointment_booking.md"
        original = path.read_text(encoding="utf-8")
        try:
            path.write_text(original + "\nA changed policy must be reindexed.\n", encoding="utf-8")
            with self.assertRaisesRegex(RuntimeError, "Stale"):
                self.index.retrieve("How do I book an appointment?")
        finally:
            path.write_text(original, encoding="utf-8")


if __name__ == "__main__":
    unittest.main()
