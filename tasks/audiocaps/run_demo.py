#!/usr/bin/env python3
"""
AudioCaps Audio-Text Retrieval Demo Script

This script demonstrates how to generate predictions for the AudioCaps task.
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
    audio_to_text = {}
    text_to_audio = {}

    for query in queries:
        qid = query["id"]
        audiocap_id = query.get("audiocap_id", "")

        # TODO: Replace this mock prediction with your actual model
        # Your model should:
        # 1. For audio-to-text: given audio, retrieve matching captions
        # 2. For text-to-audio: given caption, retrieve matching audio
        # Example:
        #   audio = load_audio(audiocap_id)
        #   captions = your_model.audio_to_text(audio, top_k=10)

        # Mock prediction: generate random rankings (as list of IDs)
        num_candidates = 100
        ranking = [f"{random.randint(0, num_candidates-1)}_caption" for _ in range(10)]

        audio_to_text[qid] = ranking
        text_to_audio[f"{qid}_caption"] = [str(random.randint(0, num_candidates-1)) for _ in range(10)]

    # Save predictions
    submission = {
        "model_name": "random-baseline",
        "audio_to_text": audio_to_text,
        "text_to_audio": text_to_audio
    }

    output_path = task_dir / "demo_predictions.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(submission, f, indent=2, ensure_ascii=False)

    print(f"Saved predictions to: {output_path}")
    print(f"Total audio_to_text predictions: {len(audio_to_text)}")
    print(f"Total text_to_audio predictions: {len(text_to_audio)}")

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
