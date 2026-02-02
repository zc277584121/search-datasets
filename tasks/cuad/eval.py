#!/usr/bin/env python3
"""
CUAD Contract Review Evaluation Script

Evaluates contract clause extraction using AUPR and F1 metrics.
"""

import argparse
import json
import re
import string
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import numpy as np


def normalize_answer(text: str) -> str:
    """Normalize answer text for comparison."""
    text = text.lower()
    text = "".join(ch for ch in text if ch not in string.punctuation)
    text = " ".join(text.split())
    return text


def get_tokens(text: str) -> List[str]:
    """Tokenize text into words."""
    if not text:
        return []
    return normalize_answer(text).split()


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


def compute_aupr(
    predictions: List[Tuple[float, bool]]
) -> float:
    """Compute Area Under Precision-Recall Curve."""
    # Sort by confidence (descending)
    sorted_preds = sorted(predictions, key=lambda x: x[0], reverse=True)

    precisions = []
    recalls = []
    total_positives = sum(1 for _, is_correct in predictions if is_correct)

    if total_positives == 0:
        return 0.0

    true_positives = 0
    for i, (conf, is_correct) in enumerate(sorted_preds):
        if is_correct:
            true_positives += 1
        precision = true_positives / (i + 1)
        recall = true_positives / total_positives
        precisions.append(precision)
        recalls.append(recall)

    # Compute AUC using trapezoidal rule
    aupr = 0.0
    for i in range(1, len(recalls)):
        aupr += (recalls[i] - recalls[i-1]) * (precisions[i] + precisions[i-1]) / 2

    return aupr


def load_ground_truth(dataset_path: Path = None) -> Dict:
    """Load ground truth from local file or HuggingFace dataset."""
    # Try loading from local ground_truth.json first
    default_path = Path(__file__).parent / "ground_truth.json"
    if default_path.exists():
        with open(default_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        # Convert to expected format
        ground_truth = {}
        for qid, item in data.items():
            answers = item.get("answers", [])
            ground_truth[qid] = {
                "answers": answers,
                "has_answer": len(answers) > 0 and any(a.strip() for a in answers)
            }
        return ground_truth

    if dataset_path and dataset_path.exists():
        with open(dataset_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        ground_truth = {}
        for qid, item in data.items():
            answers = item.get("answers", [])
            ground_truth[qid] = {
                "answers": answers,
                "has_answer": len(answers) > 0 and any(a.strip() for a in answers)
            }
        return ground_truth

    # Fallback: load from HuggingFace
    try:
        from datasets import load_dataset
        dataset = load_dataset("theatticusproject/cuad-qa", split="test")

        ground_truth = {}
        for item in dataset:
            qid = item["id"]
            answers = item["answers"]["text"]
            ground_truth[qid] = {
                "answers": answers,
                "has_answer": len(answers) > 0 and any(a.strip() for a in answers)
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
    total_f1 = 0.0
    true_positives = 0
    false_positives = 0
    false_negatives = 0

    aupr_data = []

    for qid, truth_data in ground_truth.items():
        pred_data = predictions.get(qid, {"answer": "", "confidence": 0.0})

        pred_answer = pred_data.get("answer", "")
        confidence = pred_data.get("confidence", 0.5)
        gold_answers = truth_data["answers"]
        has_answer = truth_data["has_answer"]

        # Determine if prediction indicates presence
        pred_has_answer = bool(pred_answer and pred_answer.strip())

        # Binary classification metrics
        if has_answer and pred_has_answer:
            # Compute F1 for text match
            best_f1 = max(compute_f1(pred_answer, ga) for ga in gold_answers) if gold_answers else 0.0
            total_f1 += best_f1
            if best_f1 > 0.5:  # Consider as correct if F1 > 0.5
                true_positives += 1
                aupr_data.append((confidence, True))
            else:
                false_positives += 1
                aupr_data.append((confidence, False))
        elif has_answer and not pred_has_answer:
            false_negatives += 1
            aupr_data.append((confidence, False))
        elif not has_answer and pred_has_answer:
            false_positives += 1
            aupr_data.append((confidence, False))
        else:  # Both no answer
            total_f1 += 1.0  # Perfect match on "no answer"
            aupr_data.append((1 - confidence, True))  # Lower confidence is better for no-answer

    num_samples = len(ground_truth)

    # Compute precision, recall, F1
    precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0.0
    recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    # Compute AUPR
    aupr = compute_aupr(aupr_data)

    results = {
        "task": "cuad",
        "aupr": round(100.0 * aupr, 2),
        "f1": round(100.0 * f1, 2),
        "precision": round(100.0 * precision, 2),
        "recall": round(100.0 * recall, 2),
        "avg_token_f1": round(100.0 * total_f1 / num_samples, 2),
        "num_samples": num_samples,
        "timestamp": datetime.now().isoformat()
    }

    return results


def main():
    parser = argparse.ArgumentParser(description="Evaluate CUAD predictions")
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
    print(f"AUPR: {results['aupr']}%")
    print(f"F1: {results['f1']}%")
    print(f"Precision: {results['precision']}%")
    print(f"Recall: {results['recall']}%")
    print(f"Avg Token F1: {results['avg_token_f1']}%")
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
