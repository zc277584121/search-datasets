#!/usr/bin/env python3
"""
AudioCaps Audio-Text Retrieval Evaluation Script

Evaluates bidirectional audio-text retrieval using Recall@K metrics.
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


def compute_average_precision(retrieved: List[str], relevant: Set[str]) -> float:
    """Compute average precision for a single query."""
    if not relevant:
        return 0.0

    hits = 0
    sum_precision = 0.0

    for i, item in enumerate(retrieved):
        if item in relevant:
            hits += 1
            precision_at_i = hits / (i + 1)
            sum_precision += precision_at_i

    return sum_precision / len(relevant) if len(relevant) > 0 else 0.0


def load_ground_truth(dataset_path: Path = None) -> Dict:
    """Load ground truth from local file or HuggingFace dataset."""
    # Try loading from local ground_truth.json first
    default_path = Path(__file__).parent / "ground_truth.json"
    if default_path.exists():
        with open(default_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        a2t_ground_truth = {}
        t2a_ground_truth = {}

        for audio_id, item in data.items():
            text_id = f"{audio_id}_caption"
            a2t_ground_truth[audio_id] = {text_id}
            t2a_ground_truth[text_id] = {audio_id}

        return {
            "a2t": a2t_ground_truth,
            "t2a": t2a_ground_truth
        }

    if dataset_path and dataset_path.exists():
        with open(dataset_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        a2t_ground_truth = {}
        t2a_ground_truth = {}

        for audio_id, item in data.items():
            text_id = f"{audio_id}_caption"
            a2t_ground_truth[audio_id] = {text_id}
            t2a_ground_truth[text_id] = {audio_id}

        return {
            "a2t": a2t_ground_truth,
            "t2a": t2a_ground_truth
        }

    # Fallback: load from HuggingFace
    try:
        from datasets import load_dataset
        dataset = load_dataset("AudioLLMs/audiocaps_test", split="test")

        a2t_ground_truth = {}
        t2a_ground_truth = {}

        for item in dataset:
            audio_id = str(item["id"])
            text_id = f"{audio_id}_caption"

            # Each audio may have one or more captions
            a2t_ground_truth[audio_id] = {text_id}
            t2a_ground_truth[text_id] = {audio_id}

        return {
            "a2t": a2t_ground_truth,
            "t2a": t2a_ground_truth
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
    total_ap = 0.0
    count = 0

    for query_id, relevant in ground_truth.items():
        if query_id not in predictions:
            continue

        retrieved = predictions[query_id]
        count += 1

        for k in k_values:
            recalls[k] += compute_recall_at_k(retrieved, relevant, k)

        total_rank += compute_rank(retrieved, relevant)
        total_ap += compute_average_precision(retrieved, relevant)

    if count == 0:
        return {f"r@{k}": 0.0 for k in k_values}

    results = {}
    for k in k_values:
        results[f"r@{k}"] = round(100.0 * recalls[k] / count, 2)

    results["mean_rank"] = round(total_rank / count, 2)
    results["mAP"] = round(100.0 * total_ap / count, 2)
    results["num_queries"] = count

    return results


def evaluate(
    submission: Dict,
    ground_truth: Dict
) -> Dict:
    """Evaluate both A2T and T2A retrieval."""
    results = {
        "task": "audiocaps",
        "timestamp": datetime.now().isoformat()
    }

    # Audio to Text retrieval
    if "audio_to_text" in submission:
        a2t_results = evaluate_retrieval(
            submission["audio_to_text"],
            ground_truth["a2t"]
        )
        for key, value in a2t_results.items():
            results[f"a2t_{key}"] = value

    # Text to Audio retrieval
    if "text_to_audio" in submission:
        t2a_results = evaluate_retrieval(
            submission["text_to_audio"],
            ground_truth["t2a"]
        )
        for key, value in t2a_results.items():
            results[f"t2a_{key}"] = value

    return results


def main():
    parser = argparse.ArgumentParser(description="Evaluate AudioCaps retrieval")
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

    if "a2t_r@1" in results:
        print("\nAudio to Text Retrieval:")
        print(f"  R@1: {results['a2t_r@1']}%")
        print(f"  R@5: {results['a2t_r@5']}%")
        print(f"  R@10: {results['a2t_r@10']}%")
        if "a2t_mAP" in results:
            print(f"  mAP: {results['a2t_mAP']}%")

    if "t2a_r@1" in results:
        print("\nText to Audio Retrieval:")
        print(f"  R@1: {results['t2a_r@1']}%")
        print(f"  R@5: {results['t2a_r@5']}%")
        print(f"  R@10: {results['t2a_r@10']}%")
        if "t2a_mAP" in results:
            print(f"  mAP: {results['t2a_mAP']}%")

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
