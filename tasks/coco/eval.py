#!/usr/bin/env python3
"""
COCO Image-Text Retrieval Evaluation Script

Evaluates bidirectional image-text retrieval using Recall@K metrics.
"""

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Set


def compute_recall_at_k(
    retrieved: List[str],
    relevant: Set[str],
    k: int
) -> float:
    """Compute Recall@K for a single query."""
    if not relevant:
        return 0.0

    retrieved_at_k = set(retrieved[:k])
    hits = len(retrieved_at_k & relevant)

    return hits / len(relevant)


def compute_rank(retrieved: List[str], relevant: Set[str]) -> int:
    """Compute rank of first relevant item."""
    for i, item in enumerate(retrieved):
        if item in relevant:
            return i + 1
    return len(retrieved) + 1


def load_ground_truth(dataset_path: Path = None) -> Dict:
    """Load ground truth from dataset."""
    try:
        from datasets import load_dataset
        dataset = load_dataset("nlphuji/mscoco_2014_5k_test", split="test")

        # Build image to text mapping
        i2t_ground_truth = {}
        t2i_ground_truth = {}

        for item in dataset:
            image_id = str(item["image_id"])
            # Each image has multiple captions
            caption_ids = [f"{image_id}_{i}" for i in range(5)]

            i2t_ground_truth[image_id] = set(caption_ids)

            for cap_id in caption_ids:
                t2i_ground_truth[cap_id] = {image_id}

        return {
            "i2t": i2t_ground_truth,
            "t2i": t2i_ground_truth
        }
    except Exception as e:
        print(f"Error loading dataset: {e}")
        print("Please ensure the dataset is available.")
        raise


def evaluate_retrieval(
    predictions: Dict[str, List[str]],
    ground_truth: Dict[str, Set[str]],
    k_values: List[int] = [1, 5, 10]
) -> Dict:
    """Evaluate retrieval predictions."""
    recalls = {k: 0.0 for k in k_values}
    total_rank = 0
    count = 0

    for query_id, relevant in ground_truth.items():
        if query_id not in predictions:
            continue

        retrieved = predictions[query_id]
        count += 1

        for k in k_values:
            recalls[k] += compute_recall_at_k(retrieved, relevant, k)

        total_rank += compute_rank(retrieved, relevant)

    if count == 0:
        return {f"r@{k}": 0.0 for k in k_values}

    results = {}
    for k in k_values:
        results[f"r@{k}"] = round(100.0 * recalls[k] / count, 2)

    results["mean_rank"] = round(total_rank / count, 2)
    results["num_queries"] = count

    return results


def evaluate(
    submission: Dict,
    ground_truth: Dict
) -> Dict:
    """Evaluate both I2T and T2I retrieval."""
    results = {
        "task": "coco",
        "timestamp": datetime.now().isoformat()
    }

    # Image to Text retrieval
    if "image_to_text" in submission:
        i2t_results = evaluate_retrieval(
            submission["image_to_text"],
            ground_truth["i2t"]
        )
        for key, value in i2t_results.items():
            results[f"i2t_{key}"] = value

    # Text to Image retrieval
    if "text_to_image" in submission:
        t2i_results = evaluate_retrieval(
            submission["text_to_image"],
            ground_truth["t2i"]
        )
        for key, value in t2i_results.items():
            results[f"t2i_{key}"] = value

    return results


def main():
    parser = argparse.ArgumentParser(description="Evaluate COCO retrieval")
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

    model_name = submission.get("model_name", "unknown")
    print(f"Evaluating model: {model_name}")

    # Load ground truth
    dataset_path = Path(args.dataset_path) if args.dataset_path else None
    ground_truth = load_ground_truth(dataset_path)

    # Evaluate
    results = evaluate(submission, ground_truth)
    results["model_name"] = model_name

    # Print results
    print("\n" + "=" * 50)
    print("Evaluation Results")
    print("=" * 50)

    if "i2t_r@1" in results:
        print("\nImage to Text Retrieval:")
        print(f"  R@1: {results['i2t_r@1']}%")
        print(f"  R@5: {results['i2t_r@5']}%")
        print(f"  R@10: {results['i2t_r@10']}%")

    if "t2i_r@1" in results:
        print("\nText to Image Retrieval:")
        print(f"  R@1: {results['t2i_r@1']}%")
        print(f"  R@5: {results['t2i_r@5']}%")
        print(f"  R@10: {results['t2i_r@10']}%")

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
