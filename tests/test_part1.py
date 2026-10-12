"""Part 1 checks use unittest so dataset/KB checks need no dependencies."""

from collections import Counter
import random
import unittest

from dataset import (
    APPOINTMENTS, CATEGORY_WEIGHTS, FEE_RANGES, RECORD_COUNT, STATUS_WEIGHTS,
    generate_appointments, validate_appointments,
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


if __name__ == "__main__":
    unittest.main()
