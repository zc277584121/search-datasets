#!/usr/bin/env python3
"""
GitHub Issues Retrieval Demo Script (LLM-as-Judge)

This script demonstrates how to generate predictions for the GitHub Issues task.
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
        title = query.get("title", "")
        body = query.get("body", "")

        # TODO: Replace this mock prediction with your actual model
        # Your model should retrieve relevant GitHub issues
        # Example:
        #   query_text = title + " " + body
        #   retrieved = your_retriever.search(query_text, top_k=5)

        # Mock prediction: generate random retrieved issues
        retrieved = [
            {
                "title": f"Mock issue {i+1}: Related to {title[:30]}...",
                "body": f"This is a mock issue body for testing purposes.",
                "score": random.uniform(0.5, 1.0)
            }
            for i in range(3)
        ]

        predictions.append({
            "query": title,  # Use issue title as query
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
