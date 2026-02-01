#!/usr/bin/env python3
"""
MultiHop-RAG Evaluation Script

Evaluates multi-hop retrieval and question answering.
"""

import argparse
import json
import re
import string
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Set, Tuple


def normalize_answer(text: str) -> str:
    """Normalize answer text for comparison."""
    text = text.lower()
    text = "".join(ch for ch in text if ch not in string.punctuation)
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    text = " ".join(text.split())
    return text


def get_tokens(text: str) -> List[str]:
    """Tokenize text into words."""
    if not text:
        return []
    return normalize_answer(text).split()


def compute_exact_match(prediction: str, ground_truth: str) -> float:
    """Compute exact match score."""
    return float(normalize_answer(prediction) == normalize_answer(ground_truth))


def compute_f1(prediction: str, ground_truth: str) -> float:
    """Compute F1 score between prediction and ground truth."""
    pred_tokens = get_tokens(prediction)
    truth_tokens = get_tokens(ground_truth)

    if not pred_tokens and not truth_tokens:
        return 1.0
    if not pred_tokens or not truth_tokens:
        return 0.0

    common = Counter(pred_tokens) & Counter(truth_tokens)
    num_common = sum(common.values())

    if num_common == 0:
        return 0.0

    precision = num_common / len(pred_tokens)
    recall = num_common / len(truth_tokens)
    f1 = 2 * precision * recall / (precision + recall)

    return f1


def compute_retrieval_metrics(
    retrieved: List[str],
    relevant: Set[str]
) -> Tuple[float, float]:
    """Compute retrieval precision and recall."""
    if not relevant:
        return 0.0, 0.0

    retrieved_set = set(retrieved)
    hits = len(retrieved_set & relevant)

    precision = hits / len(retrieved_set) if retrieved_set else 0.0
    recall = hits / len(relevant)

    return precision, recall


def load_ground_truth(dataset_path: Path = None) -> Dict:
    """Load ground truth from dataset."""
    try:
        from datasets import load_dataset
        dataset = load_dataset("yixuantt/MultiHopRAG", split="test")

        ground_truth = {}
        for item in dataset:
            qid = item["id"]
            ground_truth[qid] = {
                "answer": item["answer"],
                "supporting_facts": set(item.get("supporting_facts", [])),
                "num_hops": item.get("num_hops", 2)
            }
        return ground_truth
    except Exception as e:
        print(f"Error loading dataset: {e}")
        print("Please ensure the dataset is available.")
        raise


def evaluate(
    predictions: Dict[str, Dict],
    ground_truth: Dict,
) -> Dict:
    """Evaluate predictions against ground truth."""
    total_em = 0.0
    total_f1 = 0.0
    total_retrieval_precision = 0.0
    total_retrieval_recall = 0.0
    total_hops = 0

    for qid, truth_data in ground_truth.items():
        pred_data = predictions.get(qid, {"answer": "", "retrieved_docs": []})

        pred_answer = pred_data.get("answer", "")
        retrieved_docs = pred_data.get("retrieved_docs", [])
        gold_answer = truth_data["answer"]
        relevant_docs = truth_data["supporting_facts"]

        # Answer metrics
        total_em += compute_exact_match(pred_answer, gold_answer)
        total_f1 += compute_f1(pred_answer, gold_answer)

        # Retrieval metrics
        if relevant_docs:
            precision, recall = compute_retrieval_metrics(retrieved_docs, relevant_docs)
            total_retrieval_precision += precision
            total_retrieval_recall += recall

        total_hops += truth_data.get("num_hops", 2)

    num_samples = len(ground_truth)

    results = {
        "task": "multihop_rag",
        "exact_match": round(100.0 * total_em / num_samples, 2),
        "f1": round(100.0 * total_f1 / num_samples, 2),
        "retrieval_recall": round(100.0 * total_retrieval_recall / num_samples, 2),
        "retrieval_precision": round(100.0 * total_retrieval_precision / num_samples, 2),
        "avg_hops": round(total_hops / num_samples, 2),
        "num_samples": num_samples,
        "timestamp": datetime.now().isoformat()
    }

    return results


def main():
    parser = argparse.ArgumentParser(description="Evaluate MultiHop-RAG predictions")
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
    print(f"Exact Match: {results['exact_match']}%")
    print(f"F1 Score: {results['f1']}%")
    print(f"Retrieval Recall: {results['retrieval_recall']}%")
    print(f"Retrieval Precision: {results['retrieval_precision']}%")
    print(f"Average Hops: {results['avg_hops']}")
    print(f"Total Samples: {results['num_samples']}")
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
