#!/usr/bin/env python3
"""
Quora Duplicate Question Retrieval Evaluation Script

Evaluates duplicate question retrieval using Recall@K, MRR, and MAP.
"""

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Set


def compute_recall_at_k(retrieved: List[str], relevant: Set[str], k: int) -> float:
    """Compute Recall@K."""
    if not relevant:
        return 0.0
    retrieved_k = set(retrieved[:k])
    return len(retrieved_k & relevant) / len(relevant)


def compute_mrr(retrieved: List[str], relevant: Set[str]) -> float:
    """Compute Mean Reciprocal Rank."""
    for i, doc_id in enumerate(retrieved):
        if doc_id in relevant:
            return 1.0 / (i + 1)
    return 0.0


def compute_map(retrieved: List[str], relevant: Set[str]) -> float:
    """Compute Average Precision."""
    if not relevant:
        return 0.0

    hits = 0
    sum_precision = 0.0

    for i, doc_id in enumerate(retrieved):
        if doc_id in relevant:
            hits += 1
            sum_precision += hits / (i + 1)

    return sum_precision / len(relevant) if relevant else 0.0


def load_ground_truth(ground_truth_path: Path = None) -> Dict:
    """Load ground truth from local file."""
    default_path = Path(__file__).parent / "ground_truth.json"
    if default_path.exists():
        with open(default_path, "r", encoding="utf-8") as f:
            return json.load(f)

    if ground_truth_path and ground_truth_path.exists():
        with open(ground_truth_path, "r", encoding="utf-8") as f:
            return json.load(f)

    raise FileNotFoundError("ground_truth.json not found")


def evaluate(
    predictions: Dict[str, List[str]],
    ground_truth: Dict,
    k_values: List[int] = [1, 5, 10]
) -> Dict:
    """Evaluate predictions against ground truth."""
    total_mrr = 0.0
    total_map = 0.0
    recalls = {k: 0.0 for k in k_values}
    count = 0

    for qid, gt_data in ground_truth.items():
        if qid not in predictions:
            continue

        retrieved = predictions[qid]
        relevant = set(gt_data.get("duplicate_ids", []))

        if not relevant:
            continue

        count += 1
        total_mrr += compute_mrr(retrieved, relevant)
        total_map += compute_map(retrieved, relevant)

        for k in k_values:
            recalls[k] += compute_recall_at_k(retrieved, relevant, k)

    if count == 0:
        return {"error": "No valid predictions found"}

    results = {
        "task": "quora_duplicate",
        "mrr": round(100.0 * total_mrr / count, 2),
        "map": round(100.0 * total_map / count, 2),
        "num_queries": count,
        "timestamp": datetime.now().isoformat()
    }

    for k in k_values:
        results[f"recall@{k}"] = round(100.0 * recalls[k] / count, 2)

    return results


def main():
    parser = argparse.ArgumentParser(description="Evaluate Quora duplicate retrieval")
    parser.add_argument(
        "--submission",
        type=str,
        required=True,
        help="Path to submission JSON file"
    )
    parser.add_argument(
        "--output",
        type=str,
        help="Path to save evaluation results"
    )
    parser.add_argument(
        "--ground-truth",
        type=str,
        help="Path to ground truth JSON file (optional)"
    )

    args = parser.parse_args()

    # Load submission
    with open(args.submission, "r", encoding="utf-8") as f:
        submission = json.load(f)

    predictions = submission.get("predictions", {})
    model_name = submission.get("model_name", "unknown")

    print(f"Evaluating model: {model_name}")
    print(f"Number of predictions: {len(predictions)}")

    # Load ground truth
    ground_truth_path = Path(args.ground_truth) if args.ground_truth else None
    ground_truth = load_ground_truth(ground_truth_path)

    # Evaluate
    results = evaluate(predictions, ground_truth)
    results["model_name"] = model_name

    # Print results
    print("\n" + "=" * 50)
    print("Evaluation Results")
    print("=" * 50)
    print(f"MRR: {results['mrr']}%")
    print(f"MAP: {results['map']}%")
    print(f"Recall@1: {results['recall@1']}%")
    print(f"Recall@5: {results['recall@5']}%")
    print(f"Recall@10: {results['recall@10']}%")
    print(f"Num Queries: {results['num_queries']}")
    print("=" * 50)

    # Save results
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        print(f"\nResults saved to: {args.output}")

    return results


if __name__ == "__main__":
    main()
