#!/usr/bin/env python3
"""
FinQA Financial Reasoning Evaluation Script

Evaluates financial numerical reasoning using execution accuracy.
"""

import argparse
import json
import re
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


def parse_number(text: str) -> Optional[float]:
    """Parse a number from text, handling percentages and formatting."""
    if not text:
        return None

    text = str(text).strip()

    # Handle percentages
    is_percent = "%" in text
    text = text.replace("%", "").replace(",", "").replace("$", "")
    text = text.replace("(", "-").replace(")", "")  # Handle negative in parentheses

    try:
        value = float(text)
        if is_percent:
            value = value / 100
        return value
    except ValueError:
        return None


def execute_program(program: str, table_data: Dict = None) -> Optional[float]:
    """Execute a FinQA program and return the result."""
    if not program:
        return None

    # Parse program steps
    steps = [s.strip() for s in program.split(",")]
    results = []

    for step in steps:
        # Parse operation and arguments
        match = re.match(r"(\w+)\((.*)\)", step)
        if not match:
            return None

        op = match.group(1)
        args_str = match.group(2)

        # Parse arguments
        args = []
        for arg in args_str.split(","):
            arg = arg.strip()
            if arg.startswith("#"):
                # Reference to previous result
                idx = int(arg[1:])
                if idx >= len(results):
                    return None
                args.append(results[idx])
            else:
                # Literal value
                val = parse_number(arg)
                if val is None:
                    return None
                args.append(val)

        # Execute operation
        if len(args) < 2:
            return None

        try:
            if op == "add":
                result = args[0] + args[1]
            elif op == "subtract":
                result = args[0] - args[1]
            elif op == "multiply":
                result = args[0] * args[1]
            elif op == "divide":
                if args[1] == 0:
                    return None
                result = args[0] / args[1]
            elif op == "exp":
                result = args[0] ** args[1]
            elif op == "greater":
                result = 1.0 if args[0] > args[1] else 0.0
            else:
                return None

            results.append(result)
        except Exception:
            return None

    return results[-1] if results else None


def compare_numbers(pred: float, truth: float, tolerance: float = 0.01) -> bool:
    """Compare two numbers with tolerance."""
    if truth == 0:
        return abs(pred) < tolerance
    return abs(pred - truth) / abs(truth) < tolerance


def load_ground_truth(dataset_path: Path = None) -> Dict:
    """Load ground truth from dataset."""
    try:
        from datasets import load_dataset
        dataset = load_dataset("dreamerdeo/finqa", split="test")

        ground_truth = {}
        for item in dataset:
            qid = item["id"]
            ground_truth[qid] = {
                "answer": item["answer"],
                "program": item.get("program", "")
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
    execution_correct = 0
    program_correct = 0

    for qid, truth_data in ground_truth.items():
        if qid not in predictions:
            continue

        pred = predictions[qid]

        # Get predicted and ground truth values
        pred_answer = pred.get("answer", "")
        truth_answer = truth_data["answer"]

        pred_num = parse_number(str(pred_answer))
        truth_num = parse_number(str(truth_answer))

        # Check execution accuracy
        if pred_num is not None and truth_num is not None:
            if compare_numbers(pred_num, truth_num):
                execution_correct += 1

        # Check program accuracy (if programs provided)
        pred_program = pred.get("program", "")
        truth_program = truth_data.get("program", "")

        if pred_program and truth_program:
            pred_result = execute_program(pred_program)
            truth_result = execute_program(truth_program)

            if pred_result is not None and truth_result is not None:
                if compare_numbers(pred_result, truth_result):
                    program_correct += 1

    num_samples = len(ground_truth)

    results = {
        "task": "finqa",
        "execution_accuracy": round(100.0 * execution_correct / num_samples, 2),
        "num_samples": num_samples,
        "timestamp": datetime.now().isoformat()
    }

    if program_correct > 0:
        results["program_accuracy"] = round(100.0 * program_correct / num_samples, 2)

    return results


def main():
    parser = argparse.ArgumentParser(description="Evaluate FinQA predictions")
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
    print(f"Execution Accuracy: {results['execution_accuracy']}%")
    if "program_accuracy" in results:
        print(f"Program Accuracy: {results['program_accuracy']}%")
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
