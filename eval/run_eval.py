#!/usr/bin/env python3
"""
Unified evaluation runner for all tasks.

Usage:
    python eval/run_eval.py --task squad2 --submission submissions/squad2/predictions.json
    python eval/run_eval.py --task chartqa --submission submissions/chartqa/predictions.json
"""

import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from eval.metrics import (
    compute_qa_metrics,
    compute_retrieval_metrics,
    compute_relaxed_accuracy,
    compute_sql_execution_match
)


# =============================================================================
# Task Configurations
# =============================================================================

TASK_CONFIGS = {
    # QA Tasks
    'squad2': {
        'type': 'qa',
        'lang': 'en',
        'dataset_path': 'datasets/text/squad2',
        'ground_truth_file': 'validation.parquet',
        'metrics': ['exact_match', 'f1']
    },
    'cmrc2018': {
        'type': 'qa',
        'lang': 'zh',
        'dataset_path': 'datasets/chinese/cmrc2018',
        'ground_truth_file': 'validation.parquet',
        'metrics': ['exact_match', 'f1']
    },

    # Multimodal Tasks
    'chartqa': {
        'type': 'relaxed_accuracy',
        'dataset_path': 'datasets/multimodal/chartqa',
        'ground_truth_file': 'test.parquet',
        'metrics': ['accuracy', 'human_accuracy', 'augmented_accuracy']
    },
    'coco': {
        'type': 'retrieval',
        'dataset_path': 'datasets/multimodal/coco_karpathy',
        'metrics': ['recall@1', 'recall@5', 'recall@10', 'mrr']
    },

    # Audio/Video Tasks
    'audiocaps': {
        'type': 'retrieval',
        'dataset_path': 'datasets/audio/audiocaps',
        'metrics': ['recall@1', 'recall@5', 'recall@10', 'mrr']
    },
    'msvd': {
        'type': 'retrieval',
        'dataset_path': 'datasets/video/msvd',
        'metrics': ['recall@1', 'recall@5', 'recall@10', 'mrr']
    },

    # Domain Tasks
    'finqa': {
        'type': 'execution_accuracy',
        'dataset_path': 'datasets/finance/finqa',
        'ground_truth_file': 'test.json',
        'metrics': ['execution_accuracy', 'program_accuracy']
    },
    'spider': {
        'type': 'sql_execution',
        'dataset_path': 'datasets/table/spider',
        'metrics': ['execution_accuracy', 'exact_match']
    },
    'cuad': {
        'type': 'qa',
        'lang': 'en',
        'dataset_path': 'datasets/document/cuad',
        'metrics': ['f1', 'precision', 'recall']
    },

    # RAG Tasks
    'multihop_rag': {
        'type': 'rag',
        'dataset_path': 'datasets/rag/multihop_rag',
        'metrics': ['retrieval_recall@10', 'answer_f1', 'answer_em']
    },
    'eli5': {
        'type': 'retrieval',
        'dataset_path': 'datasets/conversation/eli5_reddit',
        'metrics': ['mrr', 'ndcg@10', 'recall@10']
    },

    # LLM-as-Judge Tasks
    'wildchat': {
        'type': 'llm_judge',
        'judge_type': 'dialogue',
        'dataset_path': 'datasets/conversation/wildchat_10k',
        'metrics': ['mean_relevance']
    },
    'discord': {
        'type': 'llm_judge',
        'judge_type': 'retrieval',
        'dataset_path': 'datasets/conversation/discord_chat',
        'metrics': ['mean_relevance']
    },
    'github_issues': {
        'type': 'llm_judge',
        'judge_type': 'retrieval',
        'dataset_path': 'datasets/code/github_issues',
        'metrics': ['mean_relevance']
    },
    'enron': {
        'type': 'classification',
        'dataset_path': 'datasets/document/enron_mini',
        'metrics': ['accuracy', 'f1', 'precision', 'recall']
    }
}


# =============================================================================
# Evaluation Functions
# =============================================================================

def load_predictions(path: str) -> Dict[str, Any]:
    """Load predictions from JSON file."""
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)


def load_ground_truth(task: str, config: Dict) -> Optional[Dict[str, Any]]:
    """Load ground truth data for a task."""
    dataset_path = Path(config['dataset_path'])
    gt_file = config.get('ground_truth_file')

    if gt_file:
        gt_path = dataset_path / gt_file
        if gt_path.suffix == '.json':
            with open(gt_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        elif gt_path.suffix == '.parquet':
            try:
                import pandas as pd
                return pd.read_parquet(gt_path).to_dict('records')
            except ImportError:
                print("Warning: pandas/pyarrow not installed, cannot load parquet files")
                return None
    return None


def evaluate_qa(predictions: Dict, ground_truth: list, lang: str = 'en') -> Dict[str, float]:
    """Evaluate QA task."""
    # Convert ground truth to expected format
    gt_dict = {}
    for item in ground_truth:
        qid = item.get('id', item.get('question_id', str(len(gt_dict))))
        answers = item.get('answers', {})
        if isinstance(answers, dict):
            gt_dict[qid] = answers.get('text', [])
        else:
            gt_dict[qid] = answers

    return compute_qa_metrics(predictions, gt_dict, lang)


def evaluate_retrieval(predictions: Dict, config: Dict) -> Dict[str, float]:
    """Evaluate retrieval task."""
    # Predictions format: {query_id: [retrieved_ids]}
    # Need ground truth mapping
    all_retrieved = list(predictions.values())
    # For now, assume ground truth is in predictions with 'relevant' key
    all_relevant = [set(predictions.get(f'{k}_relevant', [])) for k in predictions.keys()]

    return compute_retrieval_metrics(all_retrieved, all_relevant)


def evaluate_relaxed_accuracy(predictions: Dict, ground_truth: list) -> Dict[str, float]:
    """Evaluate with relaxed accuracy (for ChartQA etc.)."""
    correct = {'human': 0, 'augmented': 0, 'total': 0}
    total = {'human': 0, 'augmented': 0, 'total': 0}

    for item in ground_truth:
        item_id = str(item.get('id', item.get('index', len(total))))
        pred = predictions.get(item_id, '')
        gold = item.get('label', item.get('answer', ''))
        source = item.get('source', 'total')

        total[source] = total.get(source, 0) + 1
        total['total'] += 1

        if compute_relaxed_accuracy(str(pred), str(gold)) > 0.5:
            correct[source] = correct.get(source, 0) + 1
            correct['total'] += 1

    results = {'accuracy': 100.0 * correct['total'] / total['total'] if total['total'] > 0 else 0}

    if total.get('human', 0) > 0:
        results['human_accuracy'] = 100.0 * correct['human'] / total['human']
    if total.get('augmented', 0) > 0:
        results['augmented_accuracy'] = 100.0 * correct['augmented'] / total['augmented']

    results['num_samples'] = total['total']
    return results


def evaluate_classification(predictions: Dict, ground_truth: list) -> Dict[str, float]:
    """Evaluate classification task (e.g., spam detection)."""
    from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

    y_true = []
    y_pred = []

    for item in ground_truth:
        item_id = str(item.get('id', item.get('message_id', len(y_true))))
        if item_id in predictions:
            y_true.append(item.get('label', 0))
            y_pred.append(predictions[item_id])

    if not y_true:
        return {'accuracy': 0, 'f1': 0, 'precision': 0, 'recall': 0}

    return {
        'accuracy': 100.0 * accuracy_score(y_true, y_pred),
        'f1': 100.0 * f1_score(y_true, y_pred, average='binary'),
        'precision': 100.0 * precision_score(y_true, y_pred, average='binary'),
        'recall': 100.0 * recall_score(y_true, y_pred, average='binary'),
        'num_samples': len(y_true)
    }


def evaluate_llm_judge(predictions: Dict, config: Dict) -> Dict[str, float]:
    """Evaluate using LLM-as-Judge."""
    from eval.llm_judge import evaluate_with_llm_judge

    judge_type = config.get('judge_type', 'retrieval')
    queries = predictions.get('queries', [])
    items = predictions.get('predictions', [])

    return evaluate_with_llm_judge(
        task_type=judge_type,
        predictions=items,
        queries=queries
    )


def run_evaluation(task: str, predictions_path: str, output_path: Optional[str] = None) -> Dict[str, Any]:
    """
    Run evaluation for a task.

    Args:
        task: Task name
        predictions_path: Path to predictions JSON file
        output_path: Optional path to save results

    Returns:
        Evaluation results dict
    """
    if task not in TASK_CONFIGS:
        raise ValueError(f"Unknown task: {task}. Available: {list(TASK_CONFIGS.keys())}")

    config = TASK_CONFIGS[task]
    predictions = load_predictions(predictions_path)
    ground_truth = load_ground_truth(task, config)

    # Run appropriate evaluation
    eval_type = config['type']

    if eval_type == 'qa':
        results = evaluate_qa(predictions, ground_truth, config.get('lang', 'en'))
    elif eval_type == 'retrieval':
        results = evaluate_retrieval(predictions, config)
    elif eval_type == 'relaxed_accuracy':
        results = evaluate_relaxed_accuracy(predictions, ground_truth)
    elif eval_type == 'classification':
        results = evaluate_classification(predictions, ground_truth)
    elif eval_type == 'llm_judge':
        results = evaluate_llm_judge(predictions, config)
    else:
        raise ValueError(f"Unknown evaluation type: {eval_type}")

    # Add metadata
    results['task'] = task
    results['timestamp'] = datetime.now().isoformat()
    results['predictions_file'] = predictions_path

    # Save results
    if output_path:
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        print(f"Results saved to: {output_path}")

    return results


# =============================================================================
# CLI
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Run evaluation for search/retrieval tasks")
    parser.add_argument('--task', required=True, choices=list(TASK_CONFIGS.keys()),
                        help='Task to evaluate')
    parser.add_argument('--submission', required=True,
                        help='Path to submission/predictions JSON file')
    parser.add_argument('--output', help='Path to save evaluation results')
    parser.add_argument('--verbose', action='store_true', help='Print detailed results')

    args = parser.parse_args()

    print(f"Evaluating task: {args.task}")
    print(f"Submission file: {args.submission}")

    try:
        results = run_evaluation(args.task, args.submission, args.output)

        print("\n" + "=" * 50)
        print("Evaluation Results")
        print("=" * 50)

        for key, value in results.items():
            if key not in ['task', 'timestamp', 'predictions_file']:
                if isinstance(value, float):
                    print(f"  {key}: {value:.2f}")
                else:
                    print(f"  {key}: {value}")

        print("=" * 50)

    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)


if __name__ == '__main__':
    main()
