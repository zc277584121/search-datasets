#!/usr/bin/env python3
"""
MultiHop-RAG Demo Script

This script demonstrates how to generate predictions for the MultiHop-RAG task.
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
        question_type = query.get("question_type", "unknown")

        # TODO: Replace this mock prediction with your actual model
        # Your model should:
        # 1. Retrieve relevant documents (multi-hop)
        # 2. Generate an answer based on retrieved documents
        # Example:
        #   retrieved_docs = your_retriever.search(question)
        #   answer = your_model.answer(question, retrieved_docs)

        # Mock prediction: generate random answer
        if "yes" in question.lower() or "no" in question.lower():
            answer = random.choice(["Yes", "No"])
        else:
            answer = f"This is a mock answer for {question_type} question."

        # Mock retrieved documents
        retrieved_docs = [f"doc_{random.randint(1, 100)}" for _ in range(3)]

        predictions[qid] = {
            "answer": answer,
            "retrieved_docs": retrieved_docs
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
