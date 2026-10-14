"""Diagnostic raw-cosine comparison; held-out labels never select deployment."""

import argparse
import hashlib
import json
from pathlib import Path
from statistics import mean
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from rag.chunking import STRATEGIES
from rag.evaluate import document_metrics
from rag.index import COLLECTION_NAMES, RAGIndex
from rag.queries import DEMO_QUERIES, TOP_K
from rag.retrieval import RETRIEVAL_POLICY

HELDOUT_PATH = ROOT / "rag" / "heldout_queries.json"
FREEZE_PATH = ROOT / "transcripts" / "retrieval_comparison_freeze.json"
REPORT_PATH = ROOT / "transcripts" / "retrieval_comparison.json"
METHODS = ("raw_cosine", "custom")
FROZEN_SOURCE_PATHS = (
    "rag/index.py", "rag/retrieval.py", "rag/generate.py",
    "rag/chunking.py", "rag/queries.py", "rag/heldout_queries.json",
)
PRESERVED_ARTIFACTS = (
    "rag/calibration.json", "rag/deployment.json", "transcripts/task4_demo.json",
    "transcripts/task5_evaluation.json",
)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_heldout(path: Path = HELDOUT_PATH) -> list[dict]:
    queries = json.loads(path.read_text(encoding="utf-8"))
    ids = [row["id"] for row in queries]
    if not queries or len(set(ids)) != len(ids):
        raise ValueError("Held-out query IDs must be nonempty and unique")
    for row in queries:
        if not row["query"].strip() or not row["relevant_parents"]:
            raise ValueError("Held-out queries need question text and parent labels")
    return queries


def freeze_inputs(index: RAGIndex, path: Path = FREEZE_PATH) -> dict:
    """Persist labels and production-source hashes before any retrieval call."""
    if path.exists():
        raise FileExistsError("Freeze already exists; it must not be overwritten after measurement")
    files = [ROOT / name for name in FROZEN_SOURCE_PATHS] + sorted((ROOT / "kb").glob("*.md"))
    frozen = {
        "stage": "frozen before first comparison measurement",
        "label_review": "Labels reviewed against the unchanged knowledge base before measurement",
        "corpus_fingerprint": index.corpus_fingerprint,
        "retrieval_fingerprint": index.retrieval_fingerprint,
        "files": {file.relative_to(ROOT).as_posix(): sha256(file) for file in files},
        "heldout_queries": load_heldout(),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    # Exclusive creation prevents a later invocation from silently replacing the freeze.
    with path.open("x", encoding="utf-8", newline="\n") as output:
        output.write(json.dumps(frozen, indent=2, sort_keys=True) + "\n")
    text_path = path.with_suffix(".txt")
    lines = [frozen["stage"], frozen["label_review"],
             "Corpus fingerprint: " + frozen["corpus_fingerprint"],
             "Retrieval fingerprint: " + frozen["retrieval_fingerprint"]]
    lines += [f"{name}: {digest}" for name, digest in frozen["files"].items()]
    lines += [f"{row['id']}: {row['query']} | labels={row['relevant_parents']}"
              for row in frozen["heldout_queries"]]
    text_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    return frozen


def verify_frozen_inputs(frozen: dict, root: Path = ROOT) -> None:
    changed = [name for name, digest in frozen["files"].items()
               if not (root / name).is_file() or sha256(root / name) != digest]
    if changed:
        raise RuntimeError("Frozen comparison inputs changed: " + ", ".join(changed))


def raw_cosine_retrieve(index: RAGIndex, query: str, strategy: str,
                        top_k: int = TOP_K) -> list[dict]:
    """Use the same collection/model without parent or component reranking."""
    # This existing path checks corpus/index freshness without custom context selection.
    anchor = index.retrieve(query, strategy=strategy, top_k=1)[0]
    collection = index.get_collection(strategy)
    result = collection.query(
        query_embeddings=index._embed([query]),
        n_results=min(top_k, collection.count()),
        include=["documents", "metadatas", "distances"],
    )
    hits = [{"chunk_id": chunk_id, "parent_id": metadata["parent_id"],
             "text": text, "distance": float(distance),
             "similarity": 1.0 - float(distance), "metadata": metadata}
            for chunk_id, text, metadata, distance in zip(
                result["ids"][0], result["documents"][0], result["metadatas"][0],
                result["distances"][0])]
    if (hits[0]["chunk_id"] != anchor["chunk_id"]
            or hits[0]["similarity"] != anchor["similarity"]):
        raise RuntimeError("Raw cosine top-k does not preserve the raw top-one anchor")
    return hits


def evaluate_group(index: RAGIndex, queries: list[dict]) -> dict:
    """Score each query independently, then take macro document-level averages."""
    rows = []
    for labelled in queries:
        row = {"id": labelled["id"], "query": labelled["query"],
               "relevant_parents": sorted(set(labelled["relevant_parents"])),
               "strategies": {}}
        for strategy in STRATEGIES:
            methods = {}
            for method in METHODS:
                hits = (raw_cosine_retrieve(index, labelled["query"], strategy)
                        if method == "raw_cosine" else
                        index.retrieve(labelled["query"], strategy=strategy, top_k=TOP_K))
                metrics = document_metrics((hit["parent_id"] for hit in hits),
                                           labelled["relevant_parents"])
                metrics["retrieved_chunk_ids"] = [hit["chunk_id"] for hit in hits]
                metrics["hits"] = [{key: hit[key] for key in
                                    ("chunk_id", "parent_id", "distance", "similarity")}
                                   for hit in hits]
                methods[method] = metrics
            row["strategies"][strategy] = methods
        rows.append(row)
    averages = {
        strategy: {method: {metric: mean(row["strategies"][strategy][method][metric]
                                        for row in rows)
                            for metric in ("precision", "recall")}
                   for method in METHODS}
        for strategy in STRATEGIES
    }
    comparisons = {}
    for strategy in STRATEGIES:
        raw, custom = (averages[strategy][method] for method in METHODS)
        deltas = {metric: custom[metric] - raw[metric] for metric in ("precision", "recall")}
        if all(delta == 0 for delta in deltas.values()):
            outcome = "tie"
        elif all(delta >= 0 for delta in deltas.values()):
            outcome = "custom higher on at least one metric and no lower on the other"
        elif all(delta <= 0 for delta in deltas.values()):
            outcome = "raw cosine higher on at least one metric and no lower on the other"
        else:
            outcome = "precision/recall trade-off"
        comparisons[strategy] = {"custom_minus_raw": deltas, "outcome": outcome}
    return {"query_count": len(rows), "queries": rows, "averages": averages,
            "comparison": comparisons}


def format_report(report: dict) -> str:
    lines = ["Diagnostic retrieval comparison; no deployment selection or tuning",
             "Corpus fingerprint: " + report["corpus_fingerprint"],
             "Retrieval fingerprint: " + report["retrieval_fingerprint"],
             "Frozen input manifest: " + report["freeze_manifest"],
             "Held-out query SHA-256: " + report["heldout_sha256"],
             "Top-k chunks: " + str(report["top_k_chunks"]),
             "Cosine similarity = 1 - Chroma cosine distance; no similarity filtering"]
    lines += [f"{method}: {policy}" for method, policy in report["policies"].items()]
    for name, group in report["groups"].items():
        lines += ["", f"Query group: {name}; queries={group['query_count']}",
                  "| Query | Collection | Raw precision | Raw recall | Custom precision | Custom recall |",
                  "|---|---|---|---|---|---|"]
        for row in group["queries"]:
            for strategy in STRATEGIES:
                raw, custom = (row["strategies"][strategy][method] for method in METHODS)
                lines.append(f"| {row['id']} | {strategy} | {raw['precision_arithmetic']} "
                             f"| {raw['recall_arithmetic']} | {custom['precision_arithmetic']} "
                             f"| {custom['recall_arithmetic']} |")
        for strategy in STRATEGIES:
            raw, custom = (group["averages"][strategy][method] for method in METHODS)
            lines.append(f"Macro average {strategy}: raw precision={raw['precision']:.6f}, "
                         f"recall={raw['recall']:.6f}; custom precision={custom['precision']:.6f}, "
                         f"recall={custom['recall']:.6f}; {group['comparison'][strategy]['outcome']}")
        for row in group["queries"]:
            lines += ["", f"{row['id']}: {row['query']}",
                      "Relevant parents: " + ", ".join(row["relevant_parents"])]
            for strategy in STRATEGIES:
                for method in METHODS:
                    metrics = row["strategies"][strategy][method]
                    lines.append(f"{strategy}/{method} parents: " + ", ".join(metrics["retrieved_parents"]))
                    lines += [f"  {hit['chunk_id']} parent={hit['parent_id']} "
                              f"distance={hit['distance']:.9f} similarity={hit['similarity']:.9f}"
                              for hit in metrics["hits"]]
    lines += ["", "Protected artifacts unchanged: " + str(report["protected_artifacts_unchanged"]),
              "Held-out results are diagnostic only and do not select deployment, calibration, or demo queries."]
    return "\n".join(lines) + "\n"


def run_comparison(index: RAGIndex, freeze_path: Path = FREEZE_PATH,
                   output_path: Path = REPORT_PATH, persist: bool = True) -> dict:
    frozen = json.loads(freeze_path.read_text(encoding="utf-8"))
    verify_frozen_inputs(frozen)
    protected_paths = {(ROOT / name).resolve()
                       for name in set(frozen["files"]) | set(PRESERVED_ARTIFACTS)}
    protected_paths.update({freeze_path.resolve(), freeze_path.with_suffix(".txt").resolve()})
    if persist and {output_path.resolve(), output_path.with_suffix(".txt").resolve()} & protected_paths:
        raise ValueError("Comparison output must not overwrite frozen inputs or protected artifacts")
    heldout = load_heldout()
    unknown = {parent for row in heldout for parent in row["relevant_parents"]} - set(index.documents)
    if unknown:
        raise ValueError("Unknown held-out parent labels: " + ", ".join(sorted(unknown)))
    protected_before = {name: sha256(ROOT / name) for name in PRESERVED_ARTIFACTS}
    groups = {"demo": evaluate_group(index, DEMO_QUERIES),
              "heldout": evaluate_group(index, heldout)}
    verify_frozen_inputs(frozen)
    protected_after = {name: sha256(ROOT / name) for name in PRESERVED_ARTIFACTS}
    if protected_before != protected_after:
        raise RuntimeError("Comparison changed calibration, deployment, or demonstration artifacts")
    report = {
        "corpus_fingerprint": index.corpus_fingerprint,
        "retrieval_fingerprint": index.retrieval_fingerprint,
        "freeze_manifest": freeze_path.relative_to(ROOT).as_posix(),
        "heldout_sha256": frozen["files"]["rag/heldout_queries.json"],
        "top_k_chunks": TOP_K,
        "collections": COLLECTION_NAMES,
        "policies": {
            "raw_cosine": "Direct collection.query top-k by whole-query cosine distance; no reranking",
            "custom": f"Existing RAGIndex.retrieve policy {RETRIEVAL_POLICY}; raw anchor plus parent/focus/component evidence selection",
        },
        "groups": groups,
        "protected_artifacts_before": protected_before,
        "protected_artifacts_after": protected_after,
        "protected_artifacts_unchanged": True,
    }
    text = format_report(report)
    if persist:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        output_path.with_suffix(".txt").write_text(text, encoding="utf-8")
    print(text, end="")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--store", type=Path, default=ROOT / ".runtime" / "chroma")
    parser.add_argument("--freeze", action="store_true", help="Freeze inputs and exit before any measurement")
    args = parser.parse_args()
    index = RAGIndex(store_path=args.store)
    if args.freeze:
        freeze_inputs(index)
    else:
        run_comparison(index)


if __name__ == "__main__":
    main()
