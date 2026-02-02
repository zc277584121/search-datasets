#!/usr/bin/env python3
"""
MSVD Video-Text Retrieval Demo Script

This script demonstrates how to generate predictions for the MSVD task.
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
    video_to_text = {}
    text_to_video = {}

    for query in queries:
        qid = query["id"]
        video_id = query.get("video_id", qid)

        # TODO: Replace this mock prediction with your actual model
        # Your model should:
        # 1. For video-to-text: given video, retrieve matching captions
        # 2. For text-to-video: given caption, retrieve matching videos
        # Example:
        #   video = load_video(video_id)  # Load from friedrichor/MSVD
        #   captions = your_model.video_to_text(video, top_k=10)

        # Mock prediction: generate random rankings (as list of IDs)
        num_candidates = 100
        # Video to text: for this video, return list of caption IDs
        video_to_text[video_id] = [f"{video_id}_{i}" for i in range(10)]
        # Text to video: for each caption, return list of video IDs
        for i in range(5):
            text_to_video[f"{video_id}_{i}"] = [f"video_{random.randint(0, num_candidates-1)}" for _ in range(10)]

    # Save predictions
    submission = {
        "model_name": "random-baseline",
        "video_to_text": video_to_text,
        "text_to_video": text_to_video
    }

    output_path = task_dir / "demo_predictions.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(submission, f, indent=2, ensure_ascii=False)

    print(f"Saved predictions to: {output_path}")
    print(f"Total video_to_text predictions: {len(video_to_text)}")
    print(f"Total text_to_video predictions: {len(text_to_video)}")

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
