#!/usr/bin/env python3
"""
CUAD Contract Understanding Demo Script

This script demonstrates how to generate predictions for the CUAD task.
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
        context = query["context"]

        # TODO: Replace this mock prediction with your actual model
        # Your model should identify relevant clause spans from the contract
        # Return a list of extracted spans, or empty list if no clause found
        # Example:
        #   spans = your_model.extract_clauses(context, question)

        # Mock prediction: extract random spans from context
        answer = ""
        if random.random() > 0.3:  # 70% chance of finding a clause
            words = context.split()
            if len(words) > 5:
                start = random.randint(0, len(words) - 5)
                length = random.randint(3, min(10, len(words) - start))
                answer = " ".join(words[start:start + length])

        predictions[qid] = {
            "answer": answer,
            "confidence": random.uniform(0.5, 1.0)
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
