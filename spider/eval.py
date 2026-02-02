#!/usr/bin/env python3
"""
Spider Text-to-SQL Evaluation Script

Evaluates SQL generation using execution accuracy.
"""

import argparse
import json
import re
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


def normalize_sql(sql: str) -> str:
    """Normalize SQL for comparison."""
    # Convert to lowercase
    sql = sql.lower().strip()
    # Remove extra whitespace
    sql = " ".join(sql.split())
    # Remove trailing semicolon
    sql = sql.rstrip(";")
    return sql


def execute_sql(sql: str, db_path: str) -> Optional[List[Tuple]]:
    """Execute SQL and return results."""
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        cursor.execute(sql)
        results = cursor.fetchall()
        conn.close()
        return results
    except Exception as e:
        return None


def compare_results(pred_results: List[Tuple], gold_results: List[Tuple]) -> bool:
    """Compare SQL execution results."""
    if pred_results is None or gold_results is None:
        return False

    # Sort for comparison (order-independent)
    try:
        pred_sorted = sorted([tuple(str(x) for x in row) for row in pred_results])
        gold_sorted = sorted([tuple(str(x) for x in row) for row in gold_results])
        return pred_sorted == gold_sorted
    except Exception:
        return False


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
        dataset = load_dataset("xlangai/spider", split="validation")

        ground_truth = {}
        for idx, item in enumerate(dataset):
            qid = str(idx)
            ground_truth[qid] = {
                "query": item["query"],
                "db_id": item["db_id"],
                "difficulty": item.get("difficulty", "unknown")
            }
        return ground_truth
    except Exception as e:
        print(f"Error loading dataset: {e}")
        print("Please ensure the dataset is available.")
        raise


def evaluate(
    predictions: Dict[str, str],
    ground_truth: Dict,
    db_dir: Optional[Path] = None
) -> Dict:
    """Evaluate predictions against ground truth."""
    total_em = 0
    total_ex = 0

    difficulty_counts = {"easy": 0, "medium": 0, "hard": 0, "extra": 0}
    difficulty_correct = {"easy": 0, "medium": 0, "hard": 0, "extra": 0}

    for qid, truth_data in ground_truth.items():
        if qid not in predictions:
            continue

        pred_sql = predictions[qid]
        gold_sql = truth_data["query"]
        difficulty = truth_data.get("difficulty", "unknown").lower()

        # Normalize difficulty name
        if "extra" in difficulty:
            diff_key = "extra"
        elif "hard" in difficulty:
            diff_key = "hard"
        elif "medium" in difficulty:
            diff_key = "medium"
        else:
            diff_key = "easy"

        difficulty_counts[diff_key] += 1

        # Exact match (normalized)
        if normalize_sql(pred_sql) == normalize_sql(gold_sql):
            total_em += 1
            difficulty_correct[diff_key] += 1
            total_ex += 1  # If EM matches, execution should also match
        elif db_dir:
            # Try execution-based comparison
            db_path = db_dir / truth_data["db_id"] / f"{truth_data['db_id']}.sqlite"
            if db_path.exists():
                pred_results = execute_sql(pred_sql, str(db_path))
                gold_results = execute_sql(gold_sql, str(db_path))
                if compare_results(pred_results, gold_results):
                    total_ex += 1
                    difficulty_correct[diff_key] += 1

    num_samples = len(ground_truth)

    results = {
        "task": "spider",
        "exact_match": round(100.0 * total_em / num_samples, 2),
        "num_samples": num_samples,
        "timestamp": datetime.now().isoformat()
    }

    if db_dir:
        results["execution_accuracy"] = round(100.0 * total_ex / num_samples, 2)

    # Add per-difficulty results
    for diff in ["easy", "medium", "hard", "extra"]:
        if difficulty_counts[diff] > 0:
            acc = round(100.0 * difficulty_correct[diff] / difficulty_counts[diff], 2)
            results[f"{diff}_accuracy"] = acc
            results[f"{diff}_count"] = difficulty_counts[diff]

    return results


def main():
    parser = argparse.ArgumentParser(description="Evaluate Spider predictions")
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
        "--db-dir",
        type=str,
        help="Path to database directory for execution-based evaluation"
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
    db_dir = Path(args.db_dir) if args.db_dir else None
    results = evaluate(predictions, ground_truth, db_dir)
    results["model_name"] = model_name

    # Print results
    print("\n" + "=" * 50)
    print("Evaluation Results")
    print("=" * 50)
    print(f"Exact Match: {results['exact_match']}%")
    if "execution_accuracy" in results:
        print(f"Execution Accuracy: {results['execution_accuracy']}%")
    if "easy_accuracy" in results:
        print(f"Easy: {results['easy_accuracy']}%")
    if "medium_accuracy" in results:
        print(f"Medium: {results['medium_accuracy']}%")
    if "hard_accuracy" in results:
        print(f"Hard: {results['hard_accuracy']}%")
    if "extra_accuracy" in results:
        print(f"Extra Hard: {results['extra_accuracy']}%")
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
