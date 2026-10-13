"""Compare chunking strategies using hand-labelled parent-document relevance."""

import argparse
import json
from pathlib import Path
from statistics import mean
from typing import Iterable

from rag.chunking import ROOT, STRATEGIES
from rag.index import RAGIndex
from rag.queries import DEMO_QUERIES, TOP_K


def document_metrics(
    retrieved_parent_ids: Iterable[str], relevant_parent_ids: Iterable[str],
) -> dict:
    """Deduplicate chunk parents before computing either document-level metric."""
    retrieved = list(dict.fromkeys(retrieved_parent_ids))
    relevant = sorted(set(relevant_parent_ids))
    intersection = sorted(set(retrieved).intersection(relevant))
    numerator = len(intersection)
    precision_denominator = len(retrieved)
    recall_denominator = len(relevant)
    precision = numerator / precision_denominator if precision_denominator else 0.0
    recall = numerator / recall_denominator if recall_denominator else 0.0
    return {
        "retrieved_parents": retrieved,
        "relevant_parents": relevant,
        "relevant_retrieved_parents": intersection,
        "numerator": numerator,
        "precision_denominator": precision_denominator,
        "recall_denominator": recall_denominator,
        "precision": precision,
        "recall": recall,
        "precision_arithmetic": (
            f"{numerator} relevant of {precision_denominator} retrieved = {precision:.6f}"),
        "recall_arithmetic": (
            f"{numerator} retrieved of {recall_denominator} relevant = {recall:.6f}"),
    }


def _select_strategy(averages: dict) -> tuple[str, str]:
    """Prefer recall, then precision, then complete-sentence boundaries on a tie."""
    fixed, sentence = averages["fixed_overlap"], averages["sentence"]
    if fixed["recall"] != sentence["recall"]:
        selected = max(STRATEGIES, key=lambda name: averages[name]["recall"])
        reason = "higher macro recall"
    elif fixed["precision"] != sentence["precision"]:
        selected = max(STRATEGIES, key=lambda name: averages[name]["precision"])
        reason = "equal macro recall and higher macro precision"
    else:
        selected = "sentence"
        reason = "equal macro recall and precision, with complete-sentence boundaries"
    return selected, reason


def _markdown_table(rows: list[dict], strategy: str, averages: dict) -> str:
    lines = [
        "| Query | Labelled parent documents | Retrieved parent documents (deduped) "
        "| Precision arithmetic | Recall arithmetic |",
        "|---|---|---|---|---|",
    ]
    for row in rows:
        metrics = row["strategies"][strategy]
        lines.append(
            f"| {row['id']} | {', '.join(metrics['relevant_parents'])} "
            f"| {', '.join(metrics['retrieved_parents']) or '(none)'} "
            f"| {metrics['precision_arithmetic']} | {metrics['recall_arithmetic']} |")
    lines.append(
        f"| Macro average | | | {averages[strategy]['precision']:.6f} "
        f"| {averages[strategy]['recall']:.6f} |")
    return "\n".join(lines)


def _print_report(report: dict) -> None:
    print("Parent-document evaluation: retrieve chunks, map parents, deduplicate, score")
    print(f"Queries: {report['query_count']}; top-k chunks: {report['top_k_chunks']}")
    print(f"Single-parent queries: {report['single_parent_query_count']}; "
          f"multi-parent queries: {report['multi_parent_query_count']}")
    print("No similarity filtering is applied during evaluation")
    print(f"Index fingerprint: {report['fingerprint']}")
    print("\n| Query | fixed_overlap precision | fixed_overlap recall "
          "| sentence precision | sentence recall |")
    print("|---|---|---|---|---|")
    for row in report["queries"]:
        fixed, sentence = row["strategies"]["fixed_overlap"], row["strategies"]["sentence"]
        print(f"| {row['id']} | {fixed['precision_arithmetic']} | {fixed['recall_arithmetic']} "
              f"| {sentence['precision_arithmetic']} | {sentence['recall_arithmetic']} |")
    fixed, sentence = report["averages"]["fixed_overlap"], report["averages"]["sentence"]
    print(f"| Macro average | {fixed['precision']:.6f} | {fixed['recall']:.6f} "
          f"| {sentence['precision']:.6f} | {sentence['recall']:.6f} |")
    for row in report["queries"]:
        print(f"\n{row['id']}: {row['query']}")
        print("Labelled relevant parents: " + ", ".join(row["relevant_parents"]))
        for strategy in STRATEGIES:
            metrics = row["strategies"][strategy]
            print(f"{strategy} retrieved chunks: " + ", ".join(metrics["retrieved_chunk_ids"]))
            print(f"{strategy} deduped parents: " + ", ".join(metrics["retrieved_parents"]))
    for strategy in STRATEGIES:
        print(f"\nMarkdown results table: {strategy}")
        print(report["markdown_tables"][strategy])
    print("\nRecommendation: " + report["recommendation"])


def run_evaluation(index: RAGIndex, persist: bool = True) -> dict:
    """Measure both collections on the shared demo queries and optionally save."""
    fingerprint = index.corpus_fingerprint
    rows = []
    for labelled_query in DEMO_QUERIES:
        row = {
            "id": labelled_query["id"],
            "query": labelled_query["query"],
            "relevant_parents": sorted(set(labelled_query["relevant_parents"])),
            "strategies": {},
        }
        for strategy in STRATEGIES:
            hits = index.retrieve(labelled_query["query"], strategy=strategy, top_k=TOP_K)
            metrics = document_metrics(
                (hit["parent_id"] for hit in hits), labelled_query["relevant_parents"])
            metrics["retrieved_chunk_ids"] = [hit["chunk_id"] for hit in hits]
            row["strategies"][strategy] = metrics
        rows.append(row)
    if index.corpus_fingerprint != fingerprint:
        raise RuntimeError("Knowledge base changed during evaluation; rerun the evaluation")
    averages = {
        strategy: {
            metric: mean(row["strategies"][strategy][metric] for row in rows)
            for metric in ("precision", "recall")
        }
        for strategy in STRATEGIES
    }
    selected, reason = _select_strategy(averages)
    fixed, sentence = averages["fixed_overlap"], averages["sentence"]
    recommendation = (
        f"Across {len(rows)} hand-labelled queries, fixed_overlap achieved macro precision "
        f"{fixed['precision']:.6f} and macro recall {fixed['recall']:.6f}, while sentence "
        f"achieved macro precision {sentence['precision']:.6f} and macro recall "
        f"{sentence['recall']:.6f}. "
        f"Deploy {selected} because it has {reason}. "
        "This small policy sample supports a local choice and does not establish universal "
        "retrieval quality."
    )
    report = {
        "fingerprint": fingerprint,
        "top_k_chunks": TOP_K,
        "query_count": len(rows),
        "single_parent_query_count": sum(len(row["relevant_parents"]) == 1 for row in rows),
        "multi_parent_query_count": sum(len(row["relevant_parents"]) > 1 for row in rows),
        "queries": rows,
        "averages": averages,
        "recommended_strategy": selected,
        "selection_reason": reason,
        "recommendation": recommendation,
        "markdown_tables": {
            strategy: _markdown_table(rows, strategy, averages) for strategy in STRATEGIES
        },
    }
    _print_report(report)
    if persist:
        report_path = ROOT / "transcripts" / "task5_evaluation.json"
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n",
                               encoding="utf-8")
        deployment = {
            "strategy": selected,
            "fingerprint": fingerprint,
            "metrics": averages,
            "selection_reason": reason,
            "top_k_chunks": TOP_K,
            "query_count": len(rows),
        }
        deployment_path = ROOT / "rag" / "deployment.json"
        deployment_path.write_text(json.dumps(deployment, indent=2, sort_keys=True) + "\n",
                                   encoding="utf-8")
        print(f"\nSaved evaluation report: {report_path.relative_to(ROOT)}")
        print(f"Saved deployment choice: {deployment_path.relative_to(ROOT)}")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--store", type=Path, default=ROOT / ".runtime" / "chroma")
    args = parser.parse_args()
    run_evaluation(RAGIndex(store_path=args.store))


if __name__ == "__main__":
    main()
