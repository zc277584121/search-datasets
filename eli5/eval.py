#!/usr/bin/env python3
"""
ELI5 QA Retrieval Evaluation Script

Evaluates document retrieval for long-form question answering using MRR and NDCG.
"""

import argparse
import json
import math
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Set


def compute_mrr(retrieved: List[str], relevant: Set[str]) -> float:
    """Compute Mean Reciprocal Rank."""
    for i, doc_id in enumerate(retrieved):
        if doc_id in relevant:
            return 1.0 / (i + 1)
    return 0.0


def compute_dcg(relevances: List[float], k: int) -> float:
    """Compute Discounted Cumulative Gain at k."""
    dcg = 0.0
    for i in range(min(k, len(relevances))):
        dcg += relevances[i] / math.log2(i + 2)  # i+2 because log2(1) = 0
    return dcg


def compute_ndcg(retrieved: List[str], relevant: Set[str], k: int) -> float:
    """Compute Normalized DCG at k."""
    # Get relevance scores for retrieved docs
    relevances = [1.0 if doc_id in relevant else 0.0 for doc_id in retrieved[:k]]

    # Compute DCG
    dcg = compute_dcg(relevances, k)

    # Compute ideal DCG
    ideal_relevances = sorted(relevances, reverse=True)
    ideal_dcg = compute_dcg(ideal_relevances, k)

    if ideal_dcg == 0:
        return 0.0

    return dcg / ideal_dcg


def compute_recall_at_k(retrieved: List[str], relevant: Set[str], k: int) -> float:
    """Compute Recall at k."""
    if not relevant:
        return 0.0

    retrieved_at_k = set(retrieved[:k])
    hits = len(retrieved_at_k & relevant)

    return hits / len(relevant)


def load_ground_truth(dataset_path: Path = None) -> Dict:
    """Load ground truth from local file or HuggingFace dataset."""
    # Try loading from local ground_truth.json first
    default_path = Path(__file__).parent / "ground_truth.json"
    if default_path.exists():
        with open(default_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        # Convert to expected format (set of relevant docs)
        ground_truth = {}
        for qid, item in data.items():
            relevant_docs = item.get("relevant_docs", [])
            if isinstance(relevant_docs, list):
                ground_truth[qid] = set(relevant_docs)
            else:
                ground_truth[qid] = set()
        return ground_truth

    if dataset_path and dataset_path.exists():
        with open(dataset_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        ground_truth = {}
        for qid, item in data.items():
            relevant_docs = item.get("relevant_docs", [])
            if isinstance(relevant_docs, list):
                ground_truth[qid] = set(relevant_docs)
            else:
                ground_truth[qid] = set()
        return ground_truth

    # Fallback: load from HuggingFace
    try:
        from datasets import load_dataset
        dataset = load_dataset("Pavithree/eli5", split="test_eli5")

        ground_truth = {}
        for item in dataset:
            qid = item["q_id"]
            # Each question has associated relevant documents
            relevant_docs = set(item.get("relevant_docs", []))
            ground_truth[qid] = relevant_docs
        return ground_truth
    except Exception as e:
        print(f"Error loading dataset: {e}")
        print("Please ensure the dataset is available.")
        raise


def evaluate(
    predictions: Dict[str, List[str]],
    ground_truth: Dict[str, Set[str]],
    k_values: List[int] = [1, 5, 10]
) -> Dict:
    """Evaluate retrieval predictions."""
    total_mrr = 0.0
    total_ndcg = {k: 0.0 for k in k_values}
    total_recall = {k: 0.0 for k in k_values}
    count = 0

    for qid, relevant in ground_truth.items():
        if qid not in predictions:
            continue

        retrieved = predictions[qid]
        count += 1

        # MRR
        total_mrr += compute_mrr(retrieved, relevant)

        # NDCG and Recall at various k
        for k in k_values:
            total_ndcg[k] += compute_ndcg(retrieved, relevant, k)
            total_recall[k] += compute_recall_at_k(retrieved, relevant, k)

    if count == 0:
        return {"error": "No valid predictions found"}

    results = {
        "task": "eli5",
        "mrr": round(100.0 * total_mrr / count, 2),
        "num_queries": count,
        "timestamp": datetime.now().isoformat()
    }

    for k in k_values:
        results[f"ndcg@{k}"] = round(100.0 * total_ndcg[k] / count, 2)
        results[f"r@{k}"] = round(100.0 * total_recall[k] / count, 2)

    return results


def main():
    parser = argparse.ArgumentParser(description="Evaluate ELI5 retrieval")
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
        "--dataset-path",
        type=str,
        help="Path to local dataset (optional)"
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
    dataset_path = Path(args.dataset_path) if args.dataset_path else None
    ground_truth = load_ground_truth(dataset_path)

    # Evaluate
    results = evaluate(predictions, ground_truth)
    results["model_name"] = model_name

    # Print results
    print("\n" + "=" * 50)
    print("Evaluation Results")
    print("=" * 50)
    print(f"MRR: {results['mrr']}%")
    print(f"NDCG@10: {results.get('ndcg@10', 'N/A')}%")
    print(f"R@1: {results.get('r@1', 'N/A')}%")
    print(f"R@5: {results.get('r@5', 'N/A')}%")
    print(f"R@10: {results.get('r@10', 'N/A')}%")
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
