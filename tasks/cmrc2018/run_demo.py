#!/usr/bin/env python3
"""
CMRC 2018 Chinese Reading Comprehension Demo Script

This script demonstrates how to generate predictions for the CMRC 2018 task.
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
        context = query["context"]
        question = query["question"]

        # TODO: Replace this mock prediction with your actual model
        # Your model should extract a Chinese answer span from the context
        # Example:
        #   answer = your_model.extract_answer(context, question)

        # Mock prediction: extract random substring from context
        if len(context) > 10:
            start = random.randint(0, len(context) - 10)
            length = random.randint(2, 20)
            answer = context[start:start + length]
        else:
            answer = context[:5] if len(context) >= 5 else context

        predictions[qid] = answer

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
