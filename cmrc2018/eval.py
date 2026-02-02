#!/usr/bin/env python3
"""
CMRC 2018 Chinese MRC Evaluation Script

Evaluates Chinese reading comprehension predictions using EM and F1 metrics.
Character-level evaluation for Chinese text.
"""

import argparse
import json
import re
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple


def normalize_chinese_answer(text: str) -> str:
    """Normalize Chinese answer text for comparison."""
    # Remove whitespace
    text = "".join(text.split())
    # Remove punctuation (both Chinese and English)
    punctuation = (
        "！？｡。＂＃＄％＆＇（）＊＋，－／：；＜＝＞＠［＼］＾＿｀｛｜｝～｟｠"
        "｢｣､、〃》「」『』【】〔〕〖〗〘〙〚〛〜〝〞〟〰〾〿–—''‛""„‟…‧﹏"
        "!\"#$%&'()*+,-./:;<=>?@[\\]^_`{|}~"
    )
    text = "".join(ch for ch in text if ch not in punctuation)
    # Convert to lowercase (for any English characters)
    text = text.lower()
    return text


def get_chinese_tokens(text: str) -> List[str]:
    """Tokenize Chinese text into characters."""
    if not text:
        return []
    return list(normalize_chinese_answer(text))


def compute_exact_match(prediction: str, ground_truth: str) -> float:
    """Compute exact match score."""
    return float(normalize_chinese_answer(prediction) == normalize_chinese_answer(ground_truth))


def compute_f1(prediction: str, ground_truth: str) -> float:
    """Compute F1 score at character level."""
    pred_chars = get_chinese_tokens(prediction)
    truth_chars = get_chinese_tokens(ground_truth)

    if not pred_chars and not truth_chars:
        return 1.0
    if not pred_chars or not truth_chars:
        return 0.0

    common = Counter(pred_chars) & Counter(truth_chars)
    num_common = sum(common.values())

    if num_common == 0:
        return 0.0

    precision = num_common / len(pred_chars)
    recall = num_common / len(truth_chars)
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


def load_ground_truth(dataset_path: Path = None) -> Dict:
    """Load ground truth from dataset."""
    try:
        from datasets import load_dataset
        dataset = load_dataset("hfl/cmrc2018", split="validation")

        ground_truth = {}
        for item in dataset:
            qid = item["id"]
            answers = item["answers"]["text"]
            ground_truth[qid] = {"answers": answers}
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

    for qid, truth_data in ground_truth.items():
        if qid not in predictions:
            pred = ""
        else:
            pred = predictions[qid]

        truths = truth_data["answers"]
        em, f1 = compute_score_for_instance(pred, truths)
        total_em += em
        total_f1 += f1

    num_samples = len(ground_truth)

    results = {
        "task": "cmrc2018",
        "exact_match": round(100.0 * total_em / num_samples, 2),
        "f1": round(100.0 * total_f1 / num_samples, 2),
        "num_samples": num_samples,
        "timestamp": datetime.now().isoformat()
    }

    return results


def main():
    parser = argparse.ArgumentParser(description="Evaluate CMRC 2018 predictions")
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
    print("评估结果 (Evaluation Results)")
    print("=" * 50)
    print(f"Exact Match: {results['exact_match']}%")
    print(f"F1 Score: {results['f1']}%")
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
