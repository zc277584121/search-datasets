"""
LLM-as-Judge evaluation for tasks without ground truth.
"""

import os
import json
from typing import List, Dict, Any, Optional
from dataclasses import dataclass


@dataclass
class JudgeResult:
    """Result from LLM judge evaluation."""
    score: float  # 1-5 scale
    explanation: str
    raw_response: str


def get_llm_client():
    """Get OpenAI client. Requires OPENAI_API_KEY environment variable."""
    try:
        from openai import OpenAI
        return OpenAI()
    except ImportError:
        raise ImportError("Please install openai: pip install openai")
    except Exception as e:
        raise RuntimeError(f"Failed to initialize OpenAI client: {e}")


# =============================================================================
# Prompt Templates
# =============================================================================

RETRIEVAL_RELEVANCE_PROMPT = """请评估以下检索结果与查询的相关性。

查询：{query}

检索结果：
{result}

评分标准（1-5分）：
1 = 完全不相关
2 = 主题略微相关但内容不匹配
3 = 部分相关
4 = 高度相关
5 = 完全匹配查询意图

请按以下格式回复：
分数：[1-5的数字]
理由：[简短解释]
"""

DIALOGUE_QUALITY_PROMPT = """请评估以下 AI 对话的质量。

对话内容：
{dialogue}

评分标准（1-5分）：
1 = 质量很差，回答不相关或有错误
2 = 质量较差，回答基本相关但不够好
3 = 质量一般，回答正确但缺乏深度
4 = 质量较好，回答准确且有帮助
5 = 质量优秀，回答专业、详细、有洞察力

请按以下格式回复：
分数：[1-5的数字]
理由：[简短解释]
"""

EMAIL_RELEVANCE_PROMPT = """请评估以下邮件与搜索查询的相关性。

搜索查询：{query}

邮件主题：{subject}
邮件内容：{content}

评分标准（1-5分）：
1 = 完全不相关
2 = 主题略微相关
3 = 部分相关
4 = 高度相关
5 = 完全匹配查询

请按以下格式回复：
分数：[1-5的数字]
理由：[简短解释]
"""

ISSUE_RELEVANCE_PROMPT = """请评估以下 GitHub Issue 与查询的相关性。

查询：{query}

Issue 标题：{title}
Issue 内容：{body}
标签：{labels}

评分标准（1-5分）：
1 = 完全不相关
2 = 主题相关但问题不同
3 = 类似问题但不是同一个
4 = 高度相关的问题
5 = 完全相同或直接相关的问题

请按以下格式回复：
分数：[1-5的数字]
理由：[简短解释]
"""


# =============================================================================
# Judge Functions
# =============================================================================

def parse_judge_response(response: str) -> JudgeResult:
    """Parse LLM judge response to extract score and explanation."""
    lines = response.strip().split('\n')
    score = 3.0  # Default score
    explanation = ""

    for line in lines:
        line = line.strip()
        if line.startswith('分数：') or line.startswith('分数:'):
            try:
                score_str = line.split('：')[-1].split(':')[-1].strip()
                score = float(score_str)
                score = max(1, min(5, score))  # Clamp to 1-5
            except:
                pass
        elif line.startswith('理由：') or line.startswith('理由:'):
            explanation = line.split('：', 1)[-1].split(':', 1)[-1].strip()

    return JudgeResult(score=score, explanation=explanation, raw_response=response)


def judge_retrieval_relevance(
    query: str,
    retrieved_items: List[str],
    client=None,
    model: str = "gpt-4o-mini",
    max_items: int = 5
) -> Dict[str, Any]:
    """
    Judge retrieval relevance using LLM.

    Args:
        query: Search query
        retrieved_items: List of retrieved item texts
        client: OpenAI client (optional, will create if not provided)
        model: Model to use for judging
        max_items: Maximum number of items to evaluate

    Returns:
        Dict with mean_score, scores, and explanations
    """
    if client is None:
        client = get_llm_client()

    results = []
    for item in retrieved_items[:max_items]:
        prompt = RETRIEVAL_RELEVANCE_PROMPT.format(
            query=query,
            result=item[:1000]  # Truncate long items
        )

        try:
            response = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=200,
                temperature=0
            )
            result = parse_judge_response(response.choices[0].message.content)
        except Exception as e:
            result = JudgeResult(score=3.0, explanation=f"Error: {e}", raw_response="")

        results.append(result)

    scores = [r.score for r in results]
    return {
        'mean_score': sum(scores) / len(scores) if scores else 0,
        'scores': scores,
        'explanations': [r.explanation for r in results]
    }


def judge_dialogue_quality(
    dialogues: List[List[Dict[str, str]]],
    client=None,
    model: str = "gpt-4o-mini",
    max_dialogues: int = 50
) -> Dict[str, Any]:
    """
    Judge dialogue quality using LLM.

    Args:
        dialogues: List of dialogues, each is a list of {role, content} dicts
        client: OpenAI client
        model: Model to use
        max_dialogues: Maximum number of dialogues to evaluate

    Returns:
        Dict with mean_score, scores
    """
    if client is None:
        client = get_llm_client()

    results = []
    for dialogue in dialogues[:max_dialogues]:
        # Format dialogue
        dialogue_text = "\n".join([
            f"{msg['role']}: {msg['content'][:300]}..."
            for msg in dialogue[:5]  # First 5 turns
        ])

        prompt = DIALOGUE_QUALITY_PROMPT.format(dialogue=dialogue_text)

        try:
            response = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=200,
                temperature=0
            )
            result = parse_judge_response(response.choices[0].message.content)
        except Exception as e:
            result = JudgeResult(score=3.0, explanation=f"Error: {e}", raw_response="")

        results.append(result)

    scores = [r.score for r in results]
    return {
        'mean_score': sum(scores) / len(scores) if scores else 0,
        'scores': scores,
        'num_evaluated': len(results)
    }


def evaluate_with_llm_judge(
    task_type: str,
    predictions: List[Dict[str, Any]],
    queries: Optional[List[str]] = None,
    client=None,
    model: str = "gpt-4o-mini"
) -> Dict[str, Any]:
    """
    Unified LLM judge evaluation interface.

    Args:
        task_type: One of 'retrieval', 'dialogue', 'email', 'issue'
        predictions: List of prediction dicts
        queries: List of queries (for retrieval tasks)
        client: OpenAI client
        model: Model to use

    Returns:
        Evaluation results dict
    """
    if client is None:
        client = get_llm_client()

    if task_type == 'retrieval':
        all_scores = []
        for query, pred in zip(queries, predictions):
            result = judge_retrieval_relevance(
                query=query,
                retrieved_items=pred.get('retrieved', []),
                client=client,
                model=model
            )
            all_scores.append(result['mean_score'])

        return {
            'mean_relevance': sum(all_scores) / len(all_scores) if all_scores else 0,
            'num_evaluated': len(all_scores)
        }

    elif task_type == 'dialogue':
        dialogues = [p.get('conversation', []) for p in predictions]
        return judge_dialogue_quality(dialogues, client, model)

    else:
        raise ValueError(f"Unknown task type: {task_type}")


# =============================================================================
# CLI
# =============================================================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="LLM-as-Judge evaluation")
    parser.add_argument("--task", required=True, choices=['retrieval', 'dialogue', 'email', 'issue'])
    parser.add_argument("--predictions", required=True, help="Path to predictions JSON file")
    parser.add_argument("--queries", help="Path to queries file (for retrieval tasks)")
    parser.add_argument("--model", default="gpt-4o-mini", help="Model to use for judging")
    parser.add_argument("--output", help="Output file for results")

    args = parser.parse_args()

    with open(args.predictions) as f:
        predictions = json.load(f)

    queries = None
    if args.queries:
        with open(args.queries) as f:
            queries = json.load(f)

    results = evaluate_with_llm_judge(
        task_type=args.task,
        predictions=predictions,
        queries=queries,
        model=args.model
    )

    print(json.dumps(results, indent=2, ensure_ascii=False))

    if args.output:
        with open(args.output, 'w') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
