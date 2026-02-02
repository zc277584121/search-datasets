#!/usr/bin/env python3
"""
AudioCaps Caption Retrieval Evaluation Script

Evaluates caption retrieval using LLM-as-Judge.
"""

import argparse
import json
import os
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

from tqdm import tqdm

# LLM evaluation prompt template
EVALUATION_PROMPT = """You are an expert evaluator for audio caption retrieval systems.

Given a query audio caption and a retrieved caption, evaluate the semantic similarity.

Query Caption: {query}

Retrieved Caption: {retrieved}

Please rate the following on a scale of 1-5:

1. Similarity (1-5): How semantically similar is the retrieved caption to the query?
   - 5: Nearly identical meaning, same audio event
   - 4: Very similar, same type of sound/event
   - 3: Somewhat similar, related sounds
   - 2: Slightly related
   - 1: Completely different

2. Sound Match (1-5): Do both captions describe the same type of sound?
   - 5: Exact same sound source
   - 4: Same category of sound
   - 3: Related sounds
   - 2: Different but co-occurring sounds
   - 1: Unrelated sounds

Respond in JSON format:
{{"similarity": <score>, "sound_match": <score>, "reasoning": "<brief explanation>"}}
"""


def evaluate_with_llm(
    query: str,
    retrieved: str,
    api_key: str,
    model: str = "gpt-4o-mini"
) -> Dict:
    """Evaluate a single retrieval using LLM."""
    try:
        import openai
        client = openai.OpenAI(api_key=api_key)

        prompt = EVALUATION_PROMPT.format(query=query, retrieved=retrieved)

        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
            max_tokens=200
        )

        result_text = response.choices[0].message.content
        # Remove markdown code blocks if present
        if "```json" in result_text:
            result_text = result_text.split("```json")[1].split("```")[0]
        elif "```" in result_text:
            result_text = result_text.split("```")[1].split("```")[0]
        result = json.loads(result_text.strip())
        return {
            "similarity": result.get("similarity", 3),
            "sound_match": result.get("sound_match", 3),
            "reasoning": result.get("reasoning", "")
        }
    except Exception as e:
        print(f"LLM evaluation error: {e}")
        return {"similarity": 3, "sound_match": 3, "reasoning": "Error in evaluation"}


def evaluate_batch(
    predictions: List[Dict],
    api_key: Optional[str] = None,
    model: str = "gpt-4o-mini",
    max_samples: int = 100
) -> Dict:
    """Evaluate a batch of predictions."""
    if not api_key:
        api_key = os.environ.get("OPENAI_API_KEY")

    if not api_key:
        print("Warning: No API key provided. Using mock evaluation.")
        total_similarity = 0
        total_sound_match = 0

        for pred in predictions[:max_samples]:
            query_words = set(pred["query"].lower().split())
            for retrieved in pred.get("retrieved", [])[:1]:
                ret_text = retrieved.get("text", "").lower()
                overlap = sum(1 for w in query_words if w in ret_text)
                similarity = min(5, 1 + overlap)
                sound_match = min(5, 1 + overlap)
                total_similarity += similarity
                total_sound_match += sound_match
                break
            else:
                total_similarity += 1
                total_sound_match += 1

        num_samples = min(len(predictions), max_samples)
        return {
            "avg_similarity": round(total_similarity / num_samples, 2),
            "avg_sound_match": round(total_sound_match / num_samples, 2),
            "high_similarity_ratio": 0.0,
            "num_queries": num_samples,
            "note": "Mock evaluation (no API key)"
        }

    # Real LLM evaluation
    total_similarity = 0
    total_sound_match = 0
    high_similarity_count = 0

    for pred in tqdm(predictions[:max_samples], desc="Evaluating"):
        query = pred["query"]
        retrieved = pred.get("retrieved", [])

        if not retrieved:
            continue

        # Evaluate top-1 retrieved result
        top_result = retrieved[0]
        ret_text = top_result.get("text", "")

        result = evaluate_with_llm(query, ret_text, api_key, model)

        total_similarity += result["similarity"]
        total_sound_match += result["sound_match"]

        if result["similarity"] >= 4:
            high_similarity_count += 1

    num_samples = min(len(predictions), max_samples)

    return {
        "avg_similarity": round(total_similarity / num_samples, 2),
        "avg_sound_match": round(total_sound_match / num_samples, 2),
        "high_similarity_ratio": round(high_similarity_count / num_samples, 2),
        "num_queries": num_samples
    }


def main():
    parser = argparse.ArgumentParser(description="Evaluate AudioCaps retrieval")
    parser.add_argument(
        "--submission",
        type=str,
        required=True,
        help="Path to submission JSON file"
    )
    parser.add_argument(
        "--output",
        type=str,
        help="Path to save evaluation results"
    )
    parser.add_argument(
        "--api-key",
        type=str,
        help="OpenAI API key for LLM evaluation"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="gpt-4o-mini",
        help="LLM model to use for evaluation"
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=100,
        help="Maximum number of samples to evaluate"
    )

    args = parser.parse_args()

    # Load submission
    with open(args.submission, "r", encoding="utf-8") as f:
        submission = json.load(f)

    predictions = submission.get("predictions", [])
    model_name = submission.get("model_name", "unknown")

    print(f"Evaluating model: {model_name}")
    print(f"Number of predictions: {len(predictions)}")

    # Evaluate
    results = evaluate_batch(
        predictions,
        api_key=args.api_key,
        model=args.model,
        max_samples=args.max_samples
    )

    results["task"] = "audiocaps"
    results["model_name"] = model_name
    results["timestamp"] = datetime.now().isoformat()

    # Print results
    print("\n" + "=" * 50)
    print("Evaluation Results")
    print("=" * 50)
    print(f"Average Similarity: {results['avg_similarity']}/5")
    print(f"Average Sound Match: {results['avg_sound_match']}/5")
    print(f"High Similarity Ratio: {results['high_similarity_ratio']}")
    print(f"Num Queries: {results['num_queries']}")
    print("=" * 50)

    # Save results
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        print(f"\nResults saved to: {args.output}")

    return results


if __name__ == "__main__":
    main()
