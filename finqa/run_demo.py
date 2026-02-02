#!/usr/bin/env python3
"""
FinQA Financial Reasoning Demo Script

This script demonstrates how to generate predictions for the FinQA task.
"""

import json
import random
from pathlib import Path


def main():
    task_dir = Path(__file__).parent

    # Load queries
    with open(task_dir / "queries.json", "r", encoding="utf-8") as f:
        data = json.load(f)

    queries = data["queries"]
    print(f"Loaded {len(queries)} queries")

    # Generate predictions
    predictions = {}

    for query in queries:
        qid = query["id"]
        question = query["question"]
        table = query.get("table", [])

        # TODO: Replace this mock prediction with your actual model
        # Your model should analyze the financial table and compute the answer
        # Optionally also generate the reasoning program
        # Example:
        #   answer, program = your_model.solve(question, table)

        # Mock prediction: generate random numerical answers
        if "percentage" in question.lower() or "change" in question.lower():
            answer = f"{random.uniform(-50, 50):.2f}%"
        else:
            answer = f"{random.uniform(100, 10000):.2f}"

        predictions[qid] = {
            "answer": answer,
            "program": ""  # Optional: reasoning program
        }

    # Save predictions
    submission = {
        "model_name": "random-baseline",
        "predictions": predictions
    }

    output_path = task_dir / "demo_predictions.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(submission, f, indent=2, ensure_ascii=False)

    print(f"Saved predictions to: {output_path}")
    print(f"Total predictions: {len(predictions)}")

    # Run evaluation
    print("\nRunning evaluation...")
    import subprocess
    result = subprocess.run(
        ["python", str(task_dir / "eval.py"), "--submission", str(output_path)],
        capture_output=True,
        text=True
    )
    print(result.stdout)
    if result.stderr:
        print("Errors:", result.stderr)


if __name__ == "__main__":
    main()
