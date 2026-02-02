#!/usr/bin/env python3
"""
Discord Chat Retrieval Demo Script (LLM-as-Judge)

This script demonstrates how to generate predictions for the Discord task.
Note: This task uses LLM-as-Judge for evaluation.
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
    predictions = []

    for query in queries:
        qid = query["id"]
        message = query.get("message", "")

        # TODO: Replace this mock prediction with your actual model
        # Your model should retrieve relevant chat messages
        # Example:
        #   retrieved = your_retriever.search(query_text, top_k=5)

        # Mock prediction: generate random retrieved messages
        retrieved = [
            {
                "text": f"Mock retrieved message {i+1} for query {qid}",
                "score": random.uniform(0.5, 1.0),
                "channel": f"channel_{random.randint(1, 10)}"
            }
            for i in range(3)
        ]

        predictions.append({
            "query": message[:100] if message else f"Query {qid}",  # Use message as query
            "retrieved": retrieved
        })

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

    # Run evaluation (LLM-as-Judge with limited samples)
    print("\nRunning evaluation (LLM-as-Judge, max 5 samples for demo)...")
    import subprocess
    result = subprocess.run(
        [
            "python", str(task_dir / "eval.py"),
            "--submission", str(output_path),
            "--max-samples", "5"  # Limit samples for demo
        ],
        capture_output=True,
        text=True
    )
    print(result.stdout)
    if result.stderr:
        print("Errors:", result.stderr)


if __name__ == "__main__":
    main()
