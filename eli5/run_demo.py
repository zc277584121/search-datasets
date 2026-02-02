#!/usr/bin/env python3
"""
ELI5 Long-form QA Retrieval Demo Script

This script demonstrates how to generate predictions for the ELI5 task.
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

        # TODO: Replace this mock prediction with your actual model
        # Your model should retrieve relevant documents for the question
        # Example:
        #   retrieved = your_retriever.search(question, top_k=10)

        # Mock prediction: generate random document IDs
        # The evaluation expects a list of document IDs (strings)
        num_docs = random.randint(3, 10)
        retrieved = [f"wiki_{random.randint(1, 10000)}" for _ in range(num_docs)]

        predictions[qid] = retrieved

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
