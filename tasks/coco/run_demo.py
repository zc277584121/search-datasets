#!/usr/bin/env python3
"""
COCO Image-Text Retrieval Demo Script

This script demonstrates how to generate predictions for the COCO task.
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
    image_to_text = {}
    text_to_image = {}

    for query in queries:
        qid = query["id"]
        image_id = query.get("image_id", qid)

        # TODO: Replace this mock prediction with your actual model
        # Your model should:
        # 1. For image-to-text: given image, retrieve matching captions
        # 2. For text-to-image: given caption, retrieve matching images
        # Example:
        #   image = load_image(image_id)  # Load from MS COCO
        #   captions = your_model.image_to_text(image, top_k=10)

        # Mock prediction: generate random rankings (as list of IDs)
        num_candidates = 100
        # Image to text: for this image, return list of caption IDs
        image_to_text[str(image_id)] = [f"{random.randint(1, num_candidates)}_{i}" for i in range(10)]
        # Text to image: for each caption, return list of image IDs
        for i in range(5):
            text_to_image[f"{image_id}_{i}"] = [str(random.randint(1, num_candidates)) for _ in range(10)]

    # Save predictions
    submission = {
        "model_name": "random-baseline",
        "image_to_text": image_to_text,
        "text_to_image": text_to_image
    }

    output_path = task_dir / "demo_predictions.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(submission, f, indent=2, ensure_ascii=False)

    print(f"Saved predictions to: {output_path}")
    print(f"Total image_to_text predictions: {len(image_to_text)}")
    print(f"Total text_to_image predictions: {len(text_to_image)}")

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
