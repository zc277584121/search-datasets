#!/usr/bin/env python3
"""
SQuAD 2.0 Reading Comprehension Demo Script

This script demonstrates how to generate predictions for the SQuAD 2.0 task.
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
        # Your model should extract an answer span from the context
        # For unanswerable questions, return empty string ""
        # Example:
        #   answer = your_model.extract_answer(context, question)

        # Mock prediction: extract random words from context
        words = context.split()
        if len(words) > 5 and random.random() > 0.2:
            start = random.randint(0, len(words) - 5)
            answer = " ".join(words[start:start + random.randint(1, 5)])
        else:
            answer = ""  # Unanswerable

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
