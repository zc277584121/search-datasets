#!/usr/bin/env python3
"""
SQuAD 2.0 Evaluation Script

Evaluates reading comprehension predictions using Exact Match and F1 metrics.
"""

import argparse
import json
import re
import string
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple


def normalize_answer(text: str) -> str:
    """Normalize answer text for comparison."""
    # Convert to lowercase
    text = text.lower()
    # Remove punctuation
    text = "".join(ch for ch in text if ch not in string.punctuation)
    # Remove articles
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    # Remove extra whitespace
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


def compute_score_for_instance(
    prediction: str, ground_truths: List[str]
) -> Tuple[float, float]:
    """Compute best EM and F1 scores across multiple ground truths."""
    best_em = 0.0
    best_f1 = 0.0

    for truth in ground_truths:
        em = compute_exact_match(prediction, truth)
        f1 = compute_f1(prediction, truth)
        best_em = max(best_em, em)
        best_f1 = max(best_f1, f1)

    return best_em, best_f1


def load_ground_truth(dataset_path: Path) -> Dict:
    """Load ground truth from dataset."""
    # Try to load from local dataset or HuggingFace
    try:
        from datasets import load_dataset
        dataset = load_dataset("rajpurkar/squad_v2", split="validation")

        ground_truth = {}
        for item in dataset:
            qid = item["id"]
            answers = item["answers"]["text"]
            is_impossible = len(answers) == 0
            ground_truth[qid] = {
                "answers": answers if answers else [""],
                "is_impossible": is_impossible
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
    total_em = 0.0
    total_f1 = 0.0
    has_answer_em = 0.0
    has_answer_f1 = 0.0
    no_answer_em = 0.0
    no_answer_f1 = 0.0

    has_answer_count = 0
    no_answer_count = 0

    for qid, truth_data in ground_truth.items():
        if qid not in predictions:
            pred = ""
        else:
            pred = predictions[qid]

        truths = truth_data["answers"]
        is_impossible = truth_data["is_impossible"]

        em, f1 = compute_score_for_instance(pred, truths)
        total_em += em
        total_f1 += f1

        if is_impossible:
            no_answer_em += em
            no_answer_f1 += f1
            no_answer_count += 1
        else:
            has_answer_em += em
            has_answer_f1 += f1
            has_answer_count += 1

    num_samples = len(ground_truth)

    results = {
        "task": "squad2",
        "exact_match": round(100.0 * total_em / num_samples, 2),
        "f1": round(100.0 * total_f1 / num_samples, 2),
        "num_samples": num_samples,
        "timestamp": datetime.now().isoformat()
    }

    if has_answer_count > 0:
        results["has_answer_em"] = round(100.0 * has_answer_em / has_answer_count, 2)
        results["has_answer_f1"] = round(100.0 * has_answer_f1 / has_answer_count, 2)
        results["has_answer_count"] = has_answer_count

    if no_answer_count > 0:
        results["no_answer_em"] = round(100.0 * no_answer_em / no_answer_count, 2)
        results["no_answer_f1"] = round(100.0 * no_answer_f1 / no_answer_count, 2)
        results["no_answer_count"] = no_answer_count

    return results


def main():
    parser = argparse.ArgumentParser(description="Evaluate SQuAD 2.0 predictions")
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
        help="Path to local dataset (optional, will download if not provided)"
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
    if "has_answer_em" in results:
        print(f"Has Answer EM: {results['has_answer_em']}%")
        print(f"Has Answer F1: {results['has_answer_f1']}%")
    if "no_answer_em" in results:
        print(f"No Answer EM: {results['no_answer_em']}%")
        print(f"No Answer F1: {results['no_answer_f1']}%")
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
