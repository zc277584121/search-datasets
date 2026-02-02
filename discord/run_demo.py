#!/usr/bin/env python3
"""
Discord Chat Retrieval Demo Script

This script demonstrates how to generate predictions for the Discord task.
"""

import json
import random
from pathlib import Path


def main():
    task_dir = Path(__file__).parent

    # Load queries
    with open(task_dir / "queries.json", "r", encoding="utf-8") as f:
        queries_data = json.load(f)

    # Load corpus
    with open(task_dir / "corpus.json", "r", encoding="utf-8") as f:
        corpus_data = json.load(f)

    queries = queries_data["queries"]
    corpus = corpus_data["documents"]
    corpus_ids = [doc["id"] for doc in corpus]

    print(f"Loaded {len(queries)} queries")
    print(f"Loaded {len(corpus)} corpus documents")

    # Generate predictions
    predictions = {}

    for query in queries:
        qid = query["id"]
        message = query.get("message", "")

        # TODO: Replace this mock prediction with your actual model
        # Your model should retrieve similar chat messages from the corpus
        # Example:
        #   retrieved_ids = your_retriever.search(message, corpus, top_k=10)

        # Mock prediction: return random corpus document IDs (excluding self)
        available_ids = [doc_id for doc_id in corpus_ids if doc_id != qid]
        retrieved_ids = random.sample(available_ids, min(10, len(available_ids)))
        predictions[qid] = retrieved_ids

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
