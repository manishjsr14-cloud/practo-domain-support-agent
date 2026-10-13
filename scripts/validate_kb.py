"""Validate user-authored fictional policies without loading an embedding model."""

from pathlib import Path
import re

from dataset import FEE_RANGES

ROOT = Path(__file__).resolve().parents[1]
TOPICS = {
    "01_appointment_booking": (r"book", r"appointment"),
    "02_cancellation_rescheduling": (r"cancel", r"reschedul"),
    "03_consultation_fees": (r"consultation", r"\bINR\b"),
    "04_insurance_claims": (r"insurance", r"claim"),
    "05_prescription_refills": (r"prescription", r"refill"),
    "06_lab_turnaround": (r"lab", r"\b(?:hours?|(?:working|business)\s+days?)\b"),
    "07_telemedicine": (r"telemedicine", r"eligib|suitab|approval|\bonly\b"),
    "08_emergency_protocol": (r"emergency", r"emergency (?:department|services)"),
    "09_patient_privacy": (r"patient|personal", r"records|contact|information|data",
                           r"mask|protect|consent|authoriz|authoris|access|fabricat"),
    "10_follow_up_discount": (r"follow[\s-]+up", r"discount"),
    "11_second_opinion": (r"second[\s-]+opinion", r"clinician"),
    "12_home_visits": (r"home[\s-]+visits?", r"eligib|\bonly\b|available|service area"),
}

STANDARD_FEES = {
    "General Medicine": 500,
    "Cardiology": 1200,
    "Dermatology": 800,
    "Pediatrics": 600,
    "Orthopedics": 1000,
}


def validate_fee_policy(fees: str) -> dict[str, int]:
    """Check the fixed new-booking schedule and preserve recorded-fee authority."""
    parsed_fees = {}
    currency = r"(?:\bINR\b|₹)\s*"
    amount = r"(\d[\d,]*)(?!(?:[\d,\-–]|\.\d))"
    for specialty, expected in STANDARD_FEES.items():
        name = re.escape(specialty)
        patterns = (
            currency + amount + rf"\s+for\s+{name}\b",
            rf"\b{name}\b\s*(?:(?:standard|new-booking|consultations?|fees?|costs?|price)\s+)*"
            rf"(?:(?:is|are)\s+|[:=]\s*)?" + currency + amount,
        )
        values = {int(value.replace(",", "")) for pattern in patterns
                  for value in re.findall(pattern, fees, flags=re.IGNORECASE)}
        if len(values) != 1:
            raise ValueError(f"Missing or conflicting standard consultation fee: {specialty}")
        value = values.pop()
        low, high = FEE_RANGES[specialty]
        if not low <= value <= high:
            raise ValueError(f"Standard fee outside dataset fee range: {specialty}")
        if value != expected:
            raise ValueError(f"Knowledge-base standard fee mismatch: {specialty}")
        parsed_fees[specialty] = value
    sentences = re.split(r"(?<=[.!?])\s+", fees)
    if not any(re.search(r"\bstandard\b", sentence, re.IGNORECASE)
               and re.search(r"\bnew[\s-]+bookings?\b", sentence, re.IGNORECASE)
               for sentence in sentences):
        raise ValueError("Standard fee schedule must explicitly apply to new bookings")
    authority_terms = (
        r"\bexisting\s+(?:appointments?|bookings?)\b", r"\brecorded\b", r"\bfees?\b",
        r"\bauthoritative\b", r"\bstandard\b", r"\b(?:differ(?:s|ent)?|vary|adjustment)\b",
    )
    authority = any(
        all(re.search(term, sentence, re.IGNORECASE) for term in authority_terms)
        and not re.search(r"\b(?:not|never)\s+(?:\w+\s+){0,2}authoritative\b",
                          sentence, re.IGNORECASE)
        for sentence in sentences
    )
    if not authority:
        raise ValueError("Existing recorded-fee authority and standard-fee distinction missing")
    return parsed_fees


def validate_kb(kb_dir: Path = ROOT / "kb") -> list[tuple[str, int]]:
    kb_dir = Path(kb_dir)
    documents = sorted(kb_dir.glob("*.md"))
    if {path.stem for path in documents} != set(TOPICS):
        raise ValueError("Knowledge base does not cover the declared topic files")
    results = []
    for path in documents:
        raw = path.read_text(encoding="utf-8-sig")
        body = " ".join(line.strip() for line in raw.splitlines()
                        if line.strip() and not line.lstrip().startswith("#"))
        sentences = [sentence for sentence in re.split(r"(?<=[.!?])\s+", body) if sentence]
        if not 2 <= len(sentences) <= 5:
            raise ValueError(f"{path.name}: sentence count outside required range")
        for concept in TOPICS[path.stem]:
            if not re.search(concept, body, re.IGNORECASE):
                raise ValueError(f"{path.name}: missing required topic content {concept}")
        if "fictional" not in raw.lower():
            raise ValueError(f"{path.name}: fictional-policy label missing")
        results.append((path.stem, len(sentences)))
    fees = (kb_dir / "03_consultation_fees.md").read_text(encoding="utf-8")
    validate_fee_policy(fees)
    return results


def main() -> None:
    results = validate_kb()
    for parent_id, sentence_count in results:
        print(f"{parent_id}: {sentence_count} sentences; topic covered")
    print(f"Documents: {len(results)}; required sentences per document: 2-5")
    for specialty, fee in STANDARD_FEES.items():
        low, high = FEE_RANGES[specialty]
        print(f"{specialty}: standard new-booking fee INR {fee}; "
              f"dataset fee range INR {low}-{high}; within range: {low <= fee <= high}")
    print("Existing appointment recorded fees remain authoritative; "
          "standard new-booking fees do not overwrite them")
    print("Knowledge-base validation: PASS")


if __name__ == "__main__":
    main()
