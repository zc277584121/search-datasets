"""
Common evaluation metrics for search and retrieval tasks.
"""

import re
import string
from collections import Counter
from typing import List, Dict, Any, Set, Optional
import numpy as np


# =============================================================================
# Text Matching Metrics (for QA tasks)
# =============================================================================

def normalize_answer_en(text: str) -> str:
    """Normalize English answer for comparison."""
    def remove_articles(s):
        return re.sub(r'\b(a|an|the)\b', ' ', s)
    def white_space_fix(s):
        return ' '.join(s.split())
    def remove_punct(s):
        return ''.join(ch for ch in s if ch not in string.punctuation)
    def lower(s):
        return s.lower()
    return white_space_fix(remove_articles(remove_punct(lower(text))))


def normalize_answer_zh(text: str) -> str:
    """Normalize Chinese answer for comparison."""
    # Remove punctuation and whitespace, keep Chinese characters and alphanumerics
    text = re.sub(r'[^\u4e00-\u9fa5a-zA-Z0-9]', '', text)
    return text.lower()


def compute_exact_match(prediction: str, ground_truth: str, lang: str = 'en') -> float:
    """
    Compute exact match score.

    Args:
        prediction: Predicted answer
        ground_truth: Ground truth answer
        lang: Language ('en' or 'zh')

    Returns:
        1.0 if exact match, 0.0 otherwise
    """
    normalize = normalize_answer_zh if lang == 'zh' else normalize_answer_en
    return float(normalize(prediction) == normalize(ground_truth))


def compute_f1(prediction: str, ground_truth: str, lang: str = 'en') -> float:
    """
    Compute token-level F1 score.

    Args:
        prediction: Predicted answer
        ground_truth: Ground truth answer
        lang: Language ('en' for word-level, 'zh' for char-level)

    Returns:
        F1 score between 0 and 1
    """
    if lang == 'zh':
        pred_tokens = list(normalize_answer_zh(prediction))
        gold_tokens = list(normalize_answer_zh(ground_truth))
    else:
        pred_tokens = normalize_answer_en(prediction).split()
        gold_tokens = normalize_answer_en(ground_truth).split()

    if not pred_tokens or not gold_tokens:
        return float(pred_tokens == gold_tokens)

    common = Counter(pred_tokens) & Counter(gold_tokens)
    num_same = sum(common.values())

    if num_same == 0:
        return 0.0

    precision = num_same / len(pred_tokens)
    recall = num_same / len(gold_tokens)
    return 2 * precision * recall / (precision + recall)


def compute_qa_metrics(
    predictions: Dict[str, str],
    ground_truths: Dict[str, List[str]],
    lang: str = 'en'
) -> Dict[str, float]:
    """
    Compute QA metrics (EM and F1) for a dataset.

    Args:
        predictions: Dict mapping question_id to predicted answer
        ground_truths: Dict mapping question_id to list of ground truth answers
        lang: Language ('en' or 'zh')

    Returns:
        Dict with 'exact_match' and 'f1' scores (0-100)
    """
    em_scores = []
    f1_scores = []

    for qid, gold_answers in ground_truths.items():
        pred = predictions.get(qid, '')

        if not gold_answers:
            # Unanswerable question
            em = float(pred == '')
            f1 = float(pred == '')
        else:
            em = max(compute_exact_match(pred, ga, lang) for ga in gold_answers)
            f1 = max(compute_f1(pred, ga, lang) for ga in gold_answers)

        em_scores.append(em)
        f1_scores.append(f1)

    return {
        'exact_match': 100.0 * sum(em_scores) / len(em_scores),
        'f1': 100.0 * sum(f1_scores) / len(f1_scores),
        'num_samples': len(em_scores)
    }


# =============================================================================
# Retrieval Metrics
# =============================================================================

def compute_recall_at_k(
    retrieved: List[Any],
    relevant: Set[Any],
    k: int
) -> float:
    """
    Compute Recall@K.

    Args:
        retrieved: List of retrieved item IDs (ordered by relevance)
        relevant: Set of relevant item IDs
        k: Number of top results to consider

    Returns:
        Recall@K score between 0 and 1
    """
    if not relevant:
        return 1.0
    retrieved_at_k = set(retrieved[:k])
    return len(retrieved_at_k & relevant) / len(relevant)


def compute_precision_at_k(
    retrieved: List[Any],
    relevant: Set[Any],
    k: int
) -> float:
    """Compute Precision@K."""
    if k == 0:
        return 0.0
    retrieved_at_k = set(retrieved[:k])
    return len(retrieved_at_k & relevant) / k


def compute_mrr(
    retrieved: List[Any],
    relevant: Set[Any]
) -> float:
    """
    Compute Mean Reciprocal Rank.

    Args:
        retrieved: List of retrieved item IDs
        relevant: Set of relevant item IDs

    Returns:
        Reciprocal rank of first relevant item (0 if none found)
    """
    for rank, item in enumerate(retrieved, 1):
        if item in relevant:
            return 1.0 / rank
    return 0.0


def compute_ndcg_at_k(
    retrieved: List[Any],
    relevance_scores: Dict[Any, float],
    k: int
) -> float:
    """
    Compute NDCG@K.

    Args:
        retrieved: List of retrieved item IDs
        relevance_scores: Dict mapping item ID to relevance score
        k: Number of top results to consider

    Returns:
        NDCG@K score between 0 and 1
    """
    # DCG
    dcg = 0.0
    for i, item in enumerate(retrieved[:k]):
        rel = relevance_scores.get(item, 0)
        dcg += rel / np.log2(i + 2)

    # Ideal DCG
    ideal_rels = sorted(relevance_scores.values(), reverse=True)[:k]
    idcg = sum(rel / np.log2(i + 2) for i, rel in enumerate(ideal_rels))

    return dcg / idcg if idcg > 0 else 0.0


def compute_retrieval_metrics(
    all_retrieved: List[List[Any]],
    all_relevant: List[Set[Any]],
    k_values: List[int] = [1, 5, 10]
) -> Dict[str, float]:
    """
    Compute retrieval metrics for a dataset.

    Args:
        all_retrieved: List of retrieved ID lists for each query
        all_relevant: List of relevant ID sets for each query
        k_values: List of K values for Recall@K

    Returns:
        Dict with recall@k, mrr scores (0-100)
    """
    results = {}

    for k in k_values:
        recalls = [compute_recall_at_k(ret, rel, k)
                   for ret, rel in zip(all_retrieved, all_relevant)]
        results[f'recall@{k}'] = 100.0 * sum(recalls) / len(recalls)

    mrrs = [compute_mrr(ret, rel) for ret, rel in zip(all_retrieved, all_relevant)]
    results['mrr'] = 100.0 * sum(mrrs) / len(mrrs)
    results['num_samples'] = len(all_retrieved)

    return results


# =============================================================================
# Numeric Accuracy Metrics (for ChartQA, FinQA, etc.)
# =============================================================================

def is_number(s: str) -> bool:
    """Check if string represents a number."""
    try:
        float(s.replace(',', '').replace('%', '').replace('$', ''))
        return True
    except:
        return False


def parse_number(s: str) -> float:
    """Parse number from string."""
    return float(s.replace(',', '').replace('%', '').replace('$', ''))


def compute_relaxed_accuracy(
    prediction: str,
    ground_truth: str,
    tolerance: float = 0.05
) -> float:
    """
    Compute relaxed accuracy (allows tolerance for numeric answers).

    Args:
        prediction: Predicted answer
        ground_truth: Ground truth answer
        tolerance: Relative tolerance for numeric comparison (default 5%)

    Returns:
        1.0 if correct, 0.0 otherwise
    """
    pred = prediction.strip()
    gold = ground_truth.strip()

    if is_number(pred) and is_number(gold):
        pred_val = parse_number(pred)
        gold_val = parse_number(gold)
        if gold_val == 0:
            return float(pred_val == 0)
        return float(abs(pred_val - gold_val) <= tolerance * abs(gold_val))
    else:
        return float(pred.lower() == gold.lower())


# =============================================================================
# SQL Execution Metrics
# =============================================================================

def compute_sql_execution_match(
    pred_results: List[tuple],
    gold_results: List[tuple]
) -> float:
    """
    Compare SQL execution results (order-independent).

    Args:
        pred_results: Results from predicted SQL
        gold_results: Results from gold SQL

    Returns:
        1.0 if results match, 0.0 otherwise
    """
    if pred_results is None:
        return 0.0
    return float(set(map(tuple, pred_results)) == set(map(tuple, gold_results)))
