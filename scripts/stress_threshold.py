"""Measure unchanged similarity thresholds on fixed, unsupported nearby topics."""

import argparse
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from rag.chunking import STRATEGIES
from rag.generate import GroundedGenerator, deployed_strategy, load_threshold
from rag.index import RAGIndex
from rag.queries import TOP_K


# Fixed before measurement; these questions are not calibration examples.
STRESS_QUERIES = (
    "Can you diagnose my chest pain?",
    "Which medicine should I take for a fever?",
    "What are the clinic's opening hours on Sundays?",
    "Which hospital in Mumbai has the shortest waiting time?",
)


def _decision(response: dict) -> str:
    return "FALLBACK" if response["fallback"] else "ANSWER"


def _counts(responses: list[dict]) -> dict[str, int]:
    fallback = sum(response["fallback"] for response in responses)
    return {"accepted": len(responses) - fallback, "fallback": fallback}


def _print_report(report: dict) -> None:
    print("Unsupported near-topic stress test; existing calibration is unchanged")
    print(f"Top-k chunks: {report['top_k']}")
    print(f"Deployed strategy: {report['deployed_strategy']}")
    print(f"Deployed threshold: {report['deployed_threshold']:.9f}")
    print("\n| Query | fixed_overlap top-1 cosine | sentence top-1 cosine "
          "| Deployed strategy | Deployed threshold | Deployed decision |")
    print("|---|---|---|---|---|---|")
    for row in report["rows"]:
        fixed = row["strategies"]["fixed_overlap"]
        sentence = row["strategies"]["sentence"]
        print(f"| {row['query']} | {fixed['top_similarity']:.9f} "
              f"| {sentence['top_similarity']:.9f} | {report['deployed_strategy']} "
              f"| {report['deployed_threshold']:.9f} | {row['deployed_decision']} |")
    for row in report["rows"]:
        print(f"\nQuestion: {row['query']}")
        for strategy in STRATEGIES:
            response = row["strategies"][strategy]
            print(f"{strategy}: top1={response['top_similarity']:.9f}; "
                  f"threshold={response['threshold']:.9f}; "
                  f"decision={_decision(response)}")
            if not response["fallback"]:
                print("Unsupported acceptance: similarity passed, but this extract "
                      "does not establish that the question is supported")
            print(f"Answer: {response['answer']}")
            print("Source chunks: " + (", ".join(response["source_chunks"]) or "(none)"))
        deployed = row["strategies"][report["deployed_strategy"]]
        print(f"Deployed answer ({row['deployed_decision']}): {deployed['answer']}")
    print("\nComputed counts: " + json.dumps(report["counts"], sort_keys=True))
    print("Calibration SHA-256 before: " + report["calibration_sha256"])
    print("Calibration SHA-256 after: " + report["calibration_sha256_after"])
    print(f"Calibration bytes unchanged: {report['calibration_unchanged']}")
    print("Accepted unsupported queries are measured limitations of the similarity-only gate")


def run_stress(
    index: RAGIndex,
    calibration_path: Path = ROOT / "rag" / "calibration.json",
) -> dict:
    """Record actual answers for both strategies without tuning or overriding."""
    calibration_path = Path(calibration_path)
    calibration_before = calibration_path.read_bytes()
    deployed = deployed_strategy(index)
    deployed_threshold = load_threshold(index, deployed, calibration_path)
    generators = {
        strategy: GroundedGenerator(index, strategy, calibration_path)
        for strategy in STRATEGIES
    }
    rows = []
    for query in STRESS_QUERIES:
        responses = {
            strategy: generators[strategy].answer(query, top_k=TOP_K)
            for strategy in STRATEGIES
        }
        deployed_response = responses[deployed]
        rows.append({
            "query": query,
            "strategies": responses,
            "deployed_fallback": deployed_response["fallback"],
            "deployed_decision": _decision(deployed_response),
        })
    calibration_after = calibration_path.read_bytes()
    report = {
        "deployed_strategy": deployed,
        "deployed_threshold": deployed_threshold,
        "index_fingerprint": index.corpus_fingerprint,
        "top_k": TOP_K,
        "calibration_sha256": hashlib.sha256(calibration_before).hexdigest(),
        "calibration_sha256_after": hashlib.sha256(calibration_after).hexdigest(),
        "calibration_unchanged": calibration_before == calibration_after,
        "rows": rows,
        "counts": {
            "deployed": _counts([row["strategies"][deployed] for row in rows]),
            "strategies": {
                strategy: _counts([row["strategies"][strategy] for row in rows])
                for strategy in STRATEGIES
            },
        },
    }
    _print_report(report)
    if not report["calibration_unchanged"]:
        raise RuntimeError("Calibration changed during the stress test; discard this run")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--store", type=Path, default=ROOT / ".runtime" / "chroma")
    parser.add_argument("--calibration", type=Path, default=ROOT / "rag" / "calibration.json")
    parser.add_argument("--output", type=Path,
                        default=ROOT / "transcripts" / "task4_near_topic.json")
    args = parser.parse_args()
    if args.output.resolve() == args.calibration.resolve():
        parser.error("--output must differ from --calibration")
    report = run_stress(RAGIndex(store_path=args.store), args.calibration)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n",
                           encoding="utf-8")
    print(f"Saved structured stress report: {args.output}")


if __name__ == "__main__":
    main()
