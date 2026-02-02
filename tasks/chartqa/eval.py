#!/usr/bin/env python3
"""
ChartQA Evaluation Script

Evaluates chart question answering predictions using relaxed accuracy.
Allows 5% tolerance for numerical answers.
"""

import argparse
import json
import re
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional, Tuple


def extract_number(text: str) -> Optional[float]:
    """Extract numerical value from text."""
    if not text:
        return None

    # Remove common symbols
    text = text.replace(",", "").replace("$", "").replace("%", "")
    text = text.replace("million", "").replace("billion", "")
    text = text.strip()

    # Try to parse as number
    try:
        return float(text)
    except ValueError:
        # Try to find a number in the text
        match = re.search(r"[-+]?\d*\.?\d+", text)
        if match:
            try:
                return float(match.group())
            except ValueError:
                return None
    return None


def normalize_text(text: str) -> str:
    """Normalize text for comparison."""
    if not text:
        return ""
    text = text.lower().strip()
    # Remove punctuation and extra spaces
    text = re.sub(r"[^\w\s]", " ", text)
    text = " ".join(text.split())
    return text


def compute_relaxed_accuracy(
    prediction: str,
    ground_truth: str,
    tolerance: float = 0.05
) -> Tuple[bool, bool]:
    """
    Compute relaxed accuracy with tolerance for numerical values.

    Returns:
        Tuple of (relaxed_correct, strict_correct)
    """
    pred_num = extract_number(prediction)
    truth_num = extract_number(ground_truth)

    # Both are numbers
    if pred_num is not None and truth_num is not None:
        if truth_num == 0:
            strict_correct = pred_num == 0
            relaxed_correct = abs(pred_num) <= tolerance
        else:
            relative_error = abs(pred_num - truth_num) / abs(truth_num)
            relaxed_correct = relative_error <= tolerance
            strict_correct = pred_num == truth_num

        return relaxed_correct, strict_correct

    # Text comparison
    pred_norm = normalize_text(prediction)
    truth_norm = normalize_text(ground_truth)
    is_correct = pred_norm == truth_norm

    return is_correct, is_correct


def load_ground_truth(dataset_path: Path = None) -> Dict:
    """Load ground truth from local file or HuggingFace dataset."""
    # Try loading from local ground_truth.json first
    default_path = Path(__file__).parent / "ground_truth.json"
    if default_path.exists():
        with open(default_path, "r", encoding="utf-8") as f:
            return json.load(f)

    if dataset_path and dataset_path.exists():
        with open(dataset_path, "r", encoding="utf-8") as f:
            return json.load(f)

    # Fallback: load from HuggingFace
    try:
        from datasets import load_dataset
        dataset = load_dataset("HuggingFaceM4/ChartQA", split="test")

        ground_truth = {}
        for idx, item in enumerate(dataset):
            qid = str(idx)
            ground_truth[qid] = {
                "answer": item["answer"],
                "type": item.get("type", "unknown")  # human or augmented
            }
        return ground_truth
    except Exception as e:
        print(f"Error loading dataset: {e}")
        print("Please ensure the dataset is available.")
        raise


def evaluate(
    predictions: Dict[str, str],
    ground_truth: Dict,
) -> Dict:
    """Evaluate predictions against ground truth."""
    total_relaxed = 0
    total_strict = 0
    human_relaxed = 0
    human_count = 0
    augmented_relaxed = 0
    augmented_count = 0

    for qid, truth_data in ground_truth.items():
        pred = predictions.get(qid, "")
        truth = truth_data["answer"]
        q_type = truth_data.get("type", "unknown")

        relaxed_correct, strict_correct = compute_relaxed_accuracy(pred, truth)

        total_relaxed += int(relaxed_correct)
        total_strict += int(strict_correct)

        if q_type == "human":
            human_relaxed += int(relaxed_correct)
            human_count += 1
        elif q_type == "augmented":
            augmented_relaxed += int(relaxed_correct)
            augmented_count += 1

    num_samples = len(ground_truth)

    results = {
        "task": "chartqa",
        "relaxed_accuracy": round(100.0 * total_relaxed / num_samples, 2),
        "strict_accuracy": round(100.0 * total_strict / num_samples, 2),
        "num_samples": num_samples,
        "timestamp": datetime.now().isoformat()
    }

    if human_count > 0:
        results["human_accuracy"] = round(100.0 * human_relaxed / human_count, 2)
        results["human_count"] = human_count

    if augmented_count > 0:
        results["augmented_accuracy"] = round(100.0 * augmented_relaxed / augmented_count, 2)
        results["augmented_count"] = augmented_count

    return results


def main():
    parser = argparse.ArgumentParser(description="Evaluate ChartQA predictions")
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
    parser.add_argument(
        "--tolerance",
        type=float,
        default=0.05,
        help="Tolerance for numerical answers (default: 0.05)"
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
    print(f"Relaxed Accuracy: {results['relaxed_accuracy']}%")
    print(f"Strict Accuracy: {results['strict_accuracy']}%")
    if "human_accuracy" in results:
        print(f"Human Questions: {results['human_accuracy']}%")
    if "augmented_accuracy" in results:
        print(f"Augmented Questions: {results['augmented_accuracy']}%")
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
