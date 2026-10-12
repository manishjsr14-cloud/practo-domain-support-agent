"""Reproducible, fabricated appointment records; no patient identifiers."""

import argparse
from collections import Counter
import json
from pathlib import Path
import random

SEED = 42
RECORD_COUNT = 50
CATEGORY_WEIGHTS = {
    "General Medicine": 0.30,
    "Cardiology": 0.15,
    "Dermatology": 0.20,
    "Pediatrics": 0.20,
    "Orthopedics": 0.15,
}
STATUS_WEIGHTS = {
    "Completed": 0.45,
    "Scheduled": 0.25,
    "Cancelled": 0.12,
    "Rescheduled": 0.10,
    "No-Show": 0.08,
}
FEE_RANGES = {
    "General Medicine": (300, 600),
    "Pediatrics": (400, 800),
    "Dermatology": (500, 1000),
    "Orthopedics": (600, 1200),
    "Cardiology": (800, 1500),
}
FEE_ROUNDING = 50
MAX_DAYS = 30
FOLLOW_UP_PROBABILITY = 0.20


def generate_appointments(seed: int = SEED) -> list[dict]:
    """Reserve coverage first, then draw independently using a private RNG.

    The reserved rows pair categories and statuses in their declared order.
    All rows, including reserved rows, draw fee, age and follow-up randomly.
    """
    rng = random.Random(seed)
    categories, statuses = list(CATEGORY_WEIGHTS), list(STATUS_WEIGHTS)
    appointments = []
    for index in range(RECORD_COUNT):
        category = categories[index] if index < len(categories) else rng.choices(
            categories, weights=list(CATEGORY_WEIGHTS.values()), k=1
        )[0]
        status = statuses[index] if index < len(statuses) else rng.choices(
            statuses, weights=list(STATUS_WEIGHTS.values()), k=1
        )[0]
        raw_fee = rng.uniform(*FEE_RANGES[category])
        # Round a continuous uniform draw to the nearest allowed fee increment.
        fee = int(raw_fee / FEE_ROUNDING + 0.5) * FEE_ROUNDING
        appointments.append({
            "record_id": f"APT-{index + 1:03d}",
            "category": category,
            "status": status,
            "consultation_fee_inr": fee,
            "days_since_created": rng.randint(0, MAX_DAYS),
            "follow_up_required": rng.random() < FOLLOW_UP_PROBABILITY,
        })
    return appointments


APPOINTMENTS = generate_appointments()


def validate_appointments(appointments: list[dict]) -> None:
    categories = Counter(row["category"] for row in appointments)
    statuses = Counter(row["status"] for row in appointments)
    if len(appointments) != RECORD_COUNT:
        raise ValueError("Unexpected record count")
    if any(categories[name] < 3 for name in CATEGORY_WEIGHTS):
        raise ValueError("Category coverage constraint failed; regenerate by design")
    if any(statuses[name] < 1 for name in STATUS_WEIGHTS):
        raise ValueError("Status coverage constraint failed")
    fraction = sum(row["follow_up_required"] for row in appointments) / len(appointments)
    if not 0.10 <= fraction <= 0.30:
        raise ValueError("Follow-up band failed; do not edit records individually")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--export", type=Path, help="Also save the generated list as JSON")
    args = parser.parse_args()
    validate_appointments(APPOINTMENTS)
    print(f"seed={SEED}; records={len(APPOINTMENTS)}")
    print(f"fee_rounding_inr={FEE_ROUNDING}; days_since_created=uniform_integer(0,{MAX_DAYS})")
    print(f"follow_up_required=Bernoulli(p={FOLLOW_UP_PROBABILITY:.2f})")
    print("Coverage: first reserve one row per category and status, then weighted draws")
    print("Category | weight | fee range INR | count")
    for name, weight in CATEGORY_WEIGHTS.items():
        low, high = FEE_RANGES[name]
        print(f"{name} | {weight:.2f} | {low}-{high} | "
              f"{sum(row['category'] == name for row in APPOINTMENTS)}")
    print("Status | weight | count")
    for name, weight in STATUS_WEIGHTS.items():
        print(f"{name} | {weight:.2f} | {sum(row['status'] == name for row in APPOINTMENTS)}")
    follow_ups = sum(row["follow_up_required"] for row in APPOINTMENTS)
    all_fees = [value for bounds in FEE_RANGES.values() for value in bounds]
    print(f"Overall designed fee range: {min(all_fees)}-{max(all_fees)} INR")
    print(f"Follow-up required: {follow_ups}/{len(APPOINTMENTS)} = "
          f"{follow_ups / len(APPOINTMENTS):.2%} (required band: 10%-30%)")
    if args.export:
        args.export.parent.mkdir(parents=True, exist_ok=True)
        args.export.write_text(json.dumps(APPOINTMENTS, indent=2) + "\n", encoding="utf-8")
        print(f"Exported {len(APPOINTMENTS)} fabricated records to {args.export.as_posix()}")
    print("Dataset validation: PASS")


if __name__ == "__main__":
    main()
