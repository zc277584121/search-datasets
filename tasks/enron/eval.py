#!/usr/bin/env python3
"""
Enron Email Search Evaluation Script

Evaluates email retrieval using LLM-as-Judge and spam detection using F1.
"""

import argparse
import json
import os
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

# LLM evaluation prompt template
RETRIEVAL_PROMPT = """You are an expert evaluator for email retrieval systems.

Given a user query and retrieved emails, evaluate the relevance.

Query: {query}

Retrieved Emails:
{emails}

Please rate the relevance on a scale of 1-5:
- 5: Directly addresses the query, highly relevant email
- 4: Very relevant, closely related to the query
- 3: Partially relevant
- 2: Marginally relevant
- 1: Not relevant

Respond in JSON format:
{{"relevance": <score>, "reasoning": "<brief explanation>"}}
"""


def evaluate_retrieval_with_llm(
    query: str,
    emails: str,
    api_key: str,
    model: str = "gpt-4"
) -> Dict:
    """Evaluate a single retrieval using LLM."""
    try:
        import openai
        client = openai.OpenAI(api_key=api_key)

        prompt = RETRIEVAL_PROMPT.format(query=query, emails=emails)

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
            "reasoning": result.get("reasoning", "")
        }
    except Exception as e:
        print(f"LLM evaluation error: {e}")
        return {"relevance": 3, "reasoning": "Error in evaluation"}


def evaluate_retrieval(
    predictions: List[Dict],
    api_key: Optional[str] = None,
    model: str = "gpt-4",
    max_samples: int = 100
) -> Dict:
    """Evaluate retrieval predictions."""
    if not api_key:
        api_key = os.environ.get("OPENAI_API_KEY")

    if not api_key:
        print("Warning: No API key provided. Using mock evaluation.")
        total_relevance = 0

        for pred in predictions[:max_samples]:
            query_words = set(pred["query"].lower().split())
            for retrieved in pred.get("retrieved", [])[:1]:
                subject = retrieved.get("subject", "").lower()
                body = retrieved.get("body", "").lower()
                text = subject + " " + body
                overlap = sum(1 for w in query_words if w in text)
                relevance = min(5, 1 + overlap)
                total_relevance += relevance
                break
            else:
                total_relevance += 1

        num_samples = min(len(predictions), max_samples)
        return {
            "avg_relevance": round(total_relevance / num_samples, 2),
            "high_relevance_ratio": 0.0,
            "num_queries": num_samples,
            "note": "Mock evaluation (no API key)"
        }

    # Real LLM evaluation
    total_relevance = 0
    high_relevance_count = 0

    for i, pred in enumerate(predictions[:max_samples]):
        query = pred["query"]
        retrieved = pred.get("retrieved", [])

        if not retrieved:
            continue

        # Format emails for evaluation
        emails_text = ""
        for j, email in enumerate(retrieved[:3]):
            subject = email.get("subject", "No subject")
            body = email.get("body", "")[:500]
            emails_text += f"\n--- Email {j+1} ---\nSubject: {subject}\nBody: {body}\n"

        result = evaluate_retrieval_with_llm(query, emails_text, api_key, model)

        total_relevance += result["relevance"]

        if result["relevance"] >= 4:
            high_relevance_count += 1

        if (i + 1) % 10 == 0:
            print(f"Evaluated {i + 1}/{min(len(predictions), max_samples)} samples")

    num_samples = min(len(predictions), max_samples)

    return {
        "avg_relevance": round(total_relevance / num_samples, 2),
        "high_relevance_ratio": round(high_relevance_count / num_samples, 2),
        "num_queries": num_samples
    }


def evaluate_spam_detection(
    predictions: Dict[str, str],
    ground_truth: Dict[str, str]
) -> Dict:
    """Evaluate spam detection predictions."""
    true_positives = 0
    false_positives = 0
    true_negatives = 0
    false_negatives = 0

    for email_id, gold_label in ground_truth.items():
        pred_label = predictions.get(email_id, "ham")
        gold_is_spam = gold_label.lower() == "spam"
        pred_is_spam = pred_label.lower() == "spam"

        if gold_is_spam and pred_is_spam:
            true_positives += 1
        elif not gold_is_spam and pred_is_spam:
            false_positives += 1
        elif not gold_is_spam and not pred_is_spam:
            true_negatives += 1
        else:
            false_negatives += 1

    precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
    recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    accuracy = (true_positives + true_negatives) / len(ground_truth) if ground_truth else 0

    return {
        "f1": round(100.0 * f1, 2),
        "precision": round(100.0 * precision, 2),
        "recall": round(100.0 * recall, 2),
        "accuracy": round(100.0 * accuracy, 2),
        "num_samples": len(ground_truth)
    }


def load_spam_ground_truth(dataset_path: Path = None) -> Dict[str, str]:
    """Load ground truth for spam detection."""
    try:
        from datasets import load_dataset
        dataset = load_dataset("SetFit/enron_spam", split="test")

        ground_truth = {}
        for idx, item in enumerate(dataset):
            email_id = str(idx)
            label = "spam" if item["label"] == 1 else "ham"
            ground_truth[email_id] = label
        return ground_truth
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return {}


def main():
    parser = argparse.ArgumentParser(description="Evaluate Enron email search")
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
        "--task-type",
        type=str,
        choices=["retrieval", "spam_detection"],
        default="retrieval",
        help="Type of task to evaluate"
    )
    parser.add_argument(
        "--api-key",
        type=str,
        help="OpenAI API key for LLM evaluation (retrieval only)"
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

    model_name = submission.get("model_name", "unknown")
    task_type = submission.get("task_type", args.task_type)

    print(f"Evaluating model: {model_name}")
    print(f"Task type: {task_type}")

    # Evaluate based on task type
    if task_type == "retrieval":
        predictions = submission.get("predictions", [])
        print(f"Number of predictions: {len(predictions)}")
        results = evaluate_retrieval(
            predictions,
            api_key=args.api_key,
            model=args.model,
            max_samples=args.max_samples
        )
    else:  # spam_detection
        predictions = submission.get("predictions", {})
        print(f"Number of predictions: {len(predictions)}")
        ground_truth = load_spam_ground_truth()
        results = evaluate_spam_detection(predictions, ground_truth)

    results["task"] = "enron"
    results["task_type"] = task_type
    results["model_name"] = model_name
    results["timestamp"] = datetime.now().isoformat()

    # Print results
    print("\n" + "=" * 50)
    print("Evaluation Results")
    print("=" * 50)

    if task_type == "retrieval":
        print(f"Average Relevance: {results['avg_relevance']}/5")
        print(f"High Relevance Ratio: {results['high_relevance_ratio']}")
        print(f"Num Queries: {results['num_queries']}")
    else:
        print(f"F1: {results['f1']}%")
        print(f"Precision: {results['precision']}%")
        print(f"Recall: {results['recall']}%")
        print(f"Accuracy: {results['accuracy']}%")
        print(f"Num Samples: {results['num_samples']}")

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
