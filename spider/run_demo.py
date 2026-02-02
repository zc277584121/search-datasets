#!/usr/bin/env python3
"""
Spider Text-to-SQL Demo Script

This script demonstrates how to generate predictions for the Spider task.
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
        db_id = query["db_id"]

        # TODO: Replace this mock prediction with your actual model
        # Your model should generate SQL query based on the question and database schema
        # Example:
        #   sql = your_model.generate_sql(question, db_id)

        # Mock prediction: generate simple SQL templates
        sql_templates = [
            f"SELECT * FROM {db_id}",
            f"SELECT COUNT(*) FROM {db_id}",
            f"SELECT id FROM {db_id} LIMIT 10",
            f"SELECT name FROM {db_id} WHERE id = 1",
        ]
        sql = random.choice(sql_templates)

        predictions[qid] = sql

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
