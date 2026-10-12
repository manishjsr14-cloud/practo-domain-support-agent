"""Validate original fictional policy documents without loading an embedding model."""

from pathlib import Path
import re

ROOT = Path(__file__).resolve().parents[1]
TOPICS = {
    "01_appointment_booking": ("book", "appointment"),
    "02_cancellation_rescheduling": ("cancel", "reschedul"),
    "03_consultation_fees": ("consultation", "INR"),
    "04_insurance_claims": ("insurance", "claim"),
    "05_prescription_refills": ("prescription", "refill"),
    "06_lab_turnaround": ("lab", "working days"),
    "07_telemedicine": ("telemedicine", "eligib"),
    "08_emergency_protocol": ("emergency", "department"),
    "09_patient_privacy": ("privacy", "data"),
    "10_follow_up_discount": ("follow-up", "discount"),
    "11_second_opinion": ("second opinion", "clinician"),
    "12_home_visits": ("home visit", "eligib"),
}


def validate_kb(kb_dir: Path = ROOT / "kb") -> list[tuple[str, int]]:
    documents = sorted(kb_dir.glob("*.md"))
    if {path.stem for path in documents} != set(TOPICS):
        raise ValueError("Knowledge base does not cover the declared topic files")
    results = []
    for path in documents:
        raw = path.read_text(encoding="utf-8")
        body = " ".join(line.strip() for line in raw.splitlines()
                        if line.strip() and not line.startswith("#"))
        sentences = re.split(r"(?<=[.!?])\s+", body)
        if not 2 <= len(sentences) <= 5:
            raise ValueError(f"{path.name}: sentence count outside required range")
        for keyword in TOPICS[path.stem]:
            if keyword.lower() not in raw.lower():
                raise ValueError(f"{path.name}: missing topic keyword {keyword}")
        if "fictional" not in raw.lower():
            raise ValueError(f"{path.name}: fictional-policy label missing")
        results.append((path.stem, len(sentences)))
    fees = (kb_dir / "03_consultation_fees.md").read_text(encoding="utf-8")
    from dataset import FEE_RANGES
    for specialty, (low, high) in FEE_RANGES.items():
        if f"{specialty} consultations cost INR {low}-{high}" not in fees:
            raise ValueError(f"Knowledge-base fee mismatch: {specialty}")
    return results


def main() -> None:
    results = validate_kb()
    for parent_id, sentence_count in results:
        print(f"{parent_id}: {sentence_count} sentences; topic covered")
    print(f"Documents: {len(results)}; required sentences per document: 2-5")
    print("All specialty fee ranges match dataset.FEE_RANGES")
    print("Knowledge-base validation: PASS")


if __name__ == "__main__":
    main()
