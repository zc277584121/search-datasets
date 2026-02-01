#!/usr/bin/env python3
"""
GitHub Issues Retrieval Evaluation Script

Evaluates issue retrieval using LLM-as-Judge.
"""

import argparse
import json
import os
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

# LLM evaluation prompt template
EVALUATION_PROMPT = """You are an expert evaluator for GitHub issue retrieval systems.

Given a user query (describing a problem or question) and retrieved GitHub issues, evaluate the relevance and usefulness.

Query: {query}

Retrieved Issues:
{issues}

Please rate the following on a scale of 1-5:

1. Relevance (1-5): How relevant are the retrieved issues to the query?
   - 5: Directly addresses the same problem/topic
   - 4: Highly relevant, related problem
   - 3: Partially relevant
   - 2: Marginally relevant
   - 1: Not relevant

2. Usefulness (1-5): How useful would these issues be for solving the query?
   - 5: Contains solution or very helpful information
   - 4: Provides good guidance
   - 3: Somewhat helpful
   - 2: Limited help
   - 1: Not useful

Respond in JSON format:
{{"relevance": <score>, "usefulness": <score>, "reasoning": "<brief explanation>"}}
"""


def evaluate_with_llm(
    query: str,
    issues: str,
    api_key: str,
    model: str = "gpt-4"
) -> Dict:
    """Evaluate a single retrieval using LLM."""
    try:
        import openai
        client = openai.OpenAI(api_key=api_key)

        prompt = EVALUATION_PROMPT.format(query=query, issues=issues)

        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
            max_tokens=200
        )

        result_text = response.choices[0].message.content
        result = json.loads(result_text)
        return {
            "relevance": result.get("relevance", 3),
            "usefulness": result.get("usefulness", 3),
            "reasoning": result.get("reasoning", "")
        }
    except Exception as e:
        print(f"LLM evaluation error: {e}")
        return {"relevance": 3, "usefulness": 3, "reasoning": "Error in evaluation"}


def evaluate_batch(
    predictions: List[Dict],
    api_key: Optional[str] = None,
    model: str = "gpt-4",
    max_samples: int = 100
) -> Dict:
    """Evaluate a batch of predictions."""
    if not api_key:
        api_key = os.environ.get("OPENAI_API_KEY")

    if not api_key:
        print("Warning: No API key provided. Using mock evaluation.")
        total_relevance = 0
        total_usefulness = 0

        for pred in predictions[:max_samples]:
            query_words = set(pred["query"].lower().split())
            for retrieved in pred.get("retrieved", [])[:1]:
                title = retrieved.get("title", "").lower()
                body = retrieved.get("body", "").lower()
                text = title + " " + body
                overlap = sum(1 for w in query_words if w in text)
                relevance = min(5, 1 + overlap)
                usefulness = min(5, 1 + overlap)
                total_relevance += relevance
                total_usefulness += usefulness
                break
            else:
                total_relevance += 1
                total_usefulness += 1

        num_samples = min(len(predictions), max_samples)
        return {
            "avg_relevance": round(total_relevance / num_samples, 2),
            "avg_usefulness": round(total_usefulness / num_samples, 2),
            "high_relevance_ratio": 0.0,
            "num_queries": num_samples,
            "note": "Mock evaluation (no API key)"
        }

    # Real LLM evaluation
    total_relevance = 0
    total_usefulness = 0
    high_relevance_count = 0

    for i, pred in enumerate(predictions[:max_samples]):
        query = pred["query"]
        retrieved = pred.get("retrieved", [])

        if not retrieved:
            continue

        # Format issues for evaluation
        issues_text = ""
        for j, issue in enumerate(retrieved[:3]):
            title = issue.get("title", "No title")
            body = issue.get("body", "No description")[:500]  # Truncate long bodies
            issues_text += f"\n--- Issue {j+1} ---\nTitle: {title}\nBody: {body}\n"

        result = evaluate_with_llm(query, issues_text, api_key, model)

        total_relevance += result["relevance"]
        total_usefulness += result["usefulness"]

        if result["relevance"] >= 4:
            high_relevance_count += 1

        if (i + 1) % 10 == 0:
            print(f"Evaluated {i + 1}/{min(len(predictions), max_samples)} samples")

    num_samples = min(len(predictions), max_samples)

    return {
        "avg_relevance": round(total_relevance / num_samples, 2),
        "avg_usefulness": round(total_usefulness / num_samples, 2),
        "high_relevance_ratio": round(high_relevance_count / num_samples, 2),
        "num_queries": num_samples
    }


def main():
    parser = argparse.ArgumentParser(description="Evaluate GitHub Issues retrieval")
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
        default="gpt-4",
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

    results["task"] = "github_issues"
    results["model_name"] = model_name
    results["timestamp"] = datetime.now().isoformat()

    # Print results
    print("\n" + "=" * 50)
    print("Evaluation Results")
    print("=" * 50)
    print(f"Average Relevance: {results['avg_relevance']}/5")
    print(f"Average Usefulness: {results['avg_usefulness']}/5")
    print(f"High Relevance Ratio: {results['high_relevance_ratio']}")
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
