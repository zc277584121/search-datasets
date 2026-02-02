#!/usr/bin/env python3
"""
Enron Email Spam Detection Evaluation Script

Evaluates spam classification using F1, Precision, Recall metrics.
"""

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Dict


def load_ground_truth(ground_truth_path: Path = None) -> Dict[str, str]:
    """Load ground truth labels."""
    if ground_truth_path and ground_truth_path.exists():
        with open(ground_truth_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return {k: v["label"] for k, v in data.items()}

    # Try loading from default location
    default_path = Path(__file__).parent / "ground_truth.json"
    if default_path.exists():
        with open(default_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return {k: v["label"] for k, v in data.items()}

    # Fallback: load from HuggingFace
    try:
        from datasets import load_dataset
        dataset = load_dataset('SetFit/enron_spam', split='test')

        ground_truth = {}
        for idx, item in enumerate(dataset):
            email_id = str(idx)
            label = "spam" if item['label'] == 1 else "ham"
            ground_truth[email_id] = label
        return ground_truth
    except Exception as e:
        print(f"Error loading dataset: {e}")
        raise


def evaluate(
    predictions: Dict[str, str],
    ground_truth: Dict[str, str]
) -> Dict:
    """Evaluate spam detection predictions."""
    true_positives = 0
    false_positives = 0
    true_negatives = 0
    false_negatives = 0

    for email_id, gold_label in ground_truth.items():
        pred_label = predictions.get(email_id, "ham")
        gold_is_spam = gold_label.lower() == "spam"
        pred_is_spam = pred_label.lower() == "spam"

        if gold_is_spam and pred_is_spam:
            true_positives += 1
        elif not gold_is_spam and pred_is_spam:
            false_positives += 1
        elif not gold_is_spam and not pred_is_spam:
            true_negatives += 1
        else:
            false_negatives += 1

    # Calculate metrics
    precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
    recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    accuracy = (true_positives + true_negatives) / len(ground_truth) if ground_truth else 0

    return {
        "f1": round(100.0 * f1, 2),
        "precision": round(100.0 * precision, 2),
        "recall": round(100.0 * recall, 2),
        "accuracy": round(100.0 * accuracy, 2),
        "true_positives": true_positives,
        "false_positives": false_positives,
        "true_negatives": true_negatives,
        "false_negatives": false_negatives,
        "num_samples": len(ground_truth)
    }


def main():
    parser = argparse.ArgumentParser(description="Evaluate Enron spam detection")
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
    results["task"] = "enron"
    results["model_name"] = model_name
    results["timestamp"] = datetime.now().isoformat()

    # Print results
    print("\n" + "=" * 50)
    print("Evaluation Results")
    print("=" * 50)
    print(f"F1: {results['f1']}%")
    print(f"Precision: {results['precision']}%")
    print(f"Recall: {results['recall']}%")
    print(f"Accuracy: {results['accuracy']}%")
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
