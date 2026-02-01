"""
Evaluation framework for search and retrieval tasks.
"""

from .metrics import (
    compute_exact_match,
    compute_f1,
    compute_qa_metrics,
    compute_recall_at_k,
    compute_precision_at_k,
    compute_mrr,
    compute_ndcg_at_k,
    compute_retrieval_metrics,
    compute_relaxed_accuracy,
    compute_sql_execution_match
)

from .llm_judge import (
    evaluate_with_llm_judge,
    judge_retrieval_relevance,
    judge_dialogue_quality
)

__all__ = [
    'compute_exact_match',
    'compute_f1',
    'compute_qa_metrics',
    'compute_recall_at_k',
    'compute_precision_at_k',
    'compute_mrr',
    'compute_ndcg_at_k',
    'compute_retrieval_metrics',
    'compute_relaxed_accuracy',
    'compute_sql_execution_match',
    'evaluate_with_llm_judge',
    'judge_retrieval_relevance',
    'judge_dialogue_quality'
]
