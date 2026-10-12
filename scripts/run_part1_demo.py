"""Show in-scope extractive answers and a held-out out-of-scope fallback."""

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from rag.generate import GroundedGenerator
from rag.index import RAGIndex
from rag.queries import DEMO_OUT_OF_SCOPE, DEMO_QUERIES, TOP_K


def run_demo(index: RAGIndex, strategy: str | None = None,
             calibration_path: Path = ROOT / "rag" / "calibration.json") -> list[dict]:
    generator = GroundedGenerator(index, strategy, calibration_path)
    print(f"MOCK_LLM=true; strategy={generator.strategy}; top_k={TOP_K}")
    results = []
    for query in [item["query"] for item in DEMO_QUERIES] + [DEMO_OUT_OF_SCOPE]:
        result = generator.answer(query)
        results.append(result)
        print(f"\nQuestion: {query}")
        print(f"top1={result['top_similarity']:.9f}; threshold={result['threshold']:.9f}; "
              f"fallback={result['fallback']}")
        print(f"Answer: {result['answer']}")
        print(f"Sources: {', '.join(result['source_chunks']) or '(none)'}")
    if any(result["fallback"] for result in results[:-1]):
        raise AssertionError("An in-scope demonstration unexpectedly fell back")
    if not results[-1]["fallback"]:
        raise AssertionError("The out-of-scope demonstration must fall back")
    print(f"\nDemo: PASS ({len(results) - 1} in-scope answers; out-of-scope fallback)")
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--strategy", choices=["fixed_overlap", "sentence"])
    parser.add_argument("--store", type=Path, default=ROOT / ".runtime" / "chroma")
    args = parser.parse_args()
    results = run_demo(RAGIndex(store_path=args.store), args.strategy)
    path = ROOT / "transcripts" / "task4_demo.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
    print(f"Saved answers and original retrieved contexts: {path.name}")


if __name__ == "__main__":
    main()
