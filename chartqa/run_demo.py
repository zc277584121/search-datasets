#!/usr/bin/env python3
"""
ChartQA Visual Reasoning Demo Script

This script demonstrates how to generate predictions for the ChartQA task.
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
        image_index = query.get("image_index", 0)

        # TODO: Replace this mock prediction with your actual model
        # Your model should analyze the chart image and answer the question
        # Example:
        #   image = load_image(image_index)  # Load from HuggingFaceM4/ChartQA
        #   answer = your_model.answer(image, question)

        # Mock prediction: generate random answers
        if "how many" in question.lower() or "what is the" in question.lower():
            answer = str(random.randint(1, 100))
        elif "percentage" in question.lower() or "%" in question:
            answer = f"{random.randint(1, 100)}%"
        elif "yes" in question.lower() or "no" in question.lower():
            answer = random.choice(["Yes", "No"])
        else:
            answer = str(random.randint(10, 500))

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
