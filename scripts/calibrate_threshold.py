"""Measure both top-one cosine clusters and persist their midpoint thresholds."""

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from rag.index import RAGIndex
from rag.queries import CALIBRATION_IN_SCOPE, CALIBRATION_OUT_OF_SCOPE

CALIBRATION_PATH = ROOT / "rag" / "calibration.json"


def choose_threshold(in_scope: list[float], out_of_scope: list[float]) -> float:
    """Refuse to invent a separating threshold when observed clusters overlap."""
    if len(in_scope) < 3 or len(out_of_scope) < 2:
        raise ValueError("Calibration requires enough labelled queries in both groups")
    lower = max(out_of_scope)
    upper = min(in_scope)
    if lower >= upper:
        raise ValueError("Calibration clusters overlap; no separating threshold exists")
    return (lower + upper) / 2


def calibrate(index: RAGIndex, output_path: Path = CALIBRATION_PATH) -> dict:
    result = {"fingerprint": index.corpus_fingerprint, "strategies": {}}
    print("Cosine similarity = 1 - Chroma cosine distance")
    for strategy in ("fixed_overlap", "sentence"):
        measurements = []
        print(f"\nCollection strategy: {strategy}")
        for label, queries in (("in_scope", CALIBRATION_IN_SCOPE),
                               ("out_of_scope", CALIBRATION_OUT_OF_SCOPE)):
            for query in queries:
                hit = index.retrieve(query, strategy=strategy, top_k=1)[0]
                entry = {"label": label, "query": query, "similarity": hit["similarity"],
                         "parent_id": hit["parent_id"]}
                measurements.append(entry)
                print(f"{label} | {query} | top1={hit['similarity']:.9f} "
                      f"| parent={hit['parent_id']}")
        positives = [row["similarity"] for row in measurements if row["label"] == "in_scope"]
        negatives = [row["similarity"] for row in measurements if row["label"] == "out_of_scope"]
        threshold = choose_threshold(positives, negatives)
        print(f"max_out={max(negatives):.9f}; min_in={min(positives):.9f}")
        print(f"threshold=(max_out + min_in)/2 = {threshold:.9f}")
        result["strategies"][strategy] = {"threshold": threshold,
                                            "measurements": measurements}
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(f"\nSaved calibration: {output_path.name}")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--store", type=Path, default=ROOT / ".runtime" / "chroma")
    args = parser.parse_args()
    calibrate(RAGIndex(store_path=args.store))


if __name__ == "__main__":
    main()
