# 任务：ELI5 长文本问答检索

## 任务描述

构建一个问答检索系统，能够为"像给五岁小孩解释一样"的问题找到最佳答案。系统需要从多个社区回答中检索和排序最相关、最易懂的答案。

## 数据集信息

| 属性 | 值 |
|------|-----|
| **名称** | ELI5 (Explain Like I'm 5) |
| **来源** | `Pavithree/eli5` |
| **语言** | 英语 |
| **许可证** | 见原始数据集 |
| **规模** | 训练集 ~216K，测试集 10K |

### 数据字段

| 字段 | 类型 | 描述 |
|------|------|------|
| `q_id` | string | 问题唯一标识符 |
| `title` | string | 问题标题 |
| `selftext` | string | 问题补充说明 |
| `answers` | dict | 包含 a_id、score、text 列表 |
| `subreddit` | string | 来源子版块 |

### 答案结构

```json
{
  "a_id": ["abc123", "def456"],
  "score": [1250, 340],
  "text": ["答案1文本...", "答案2文本..."]
}
```

score 表示 Reddit 社区投票分数，可用于质量筛选。

## 评估目标

达到：
- **检索 MRR ≥ 0.5**：最相关答案的平均倒数排名
- **ROUGE-L ≥ 0.3**：生成答案与最高分答案的重叠
- **答案排序 NDCG ≥ 0.7**：按质量排序的准确性

## 评估方法

### 检索评估

```python
import numpy as np

def compute_mrr(retrieved_ids, relevant_ids):
    """
    计算 MRR（平均倒数排名）

    参数:
        retrieved_ids: 检索结果 ID 列表
        relevant_ids: 相关答案 ID 集合
    """
    for rank, doc_id in enumerate(retrieved_ids, 1):
        if doc_id in relevant_ids:
            return 1.0 / rank
    return 0.0

def compute_ndcg(predicted_scores, true_scores, k=10):
    """
    计算 NDCG@K

    参数:
        predicted_scores: 预测的相关性分数
        true_scores: 真实的相关性分数（如 Reddit score）
    """
    # 按预测分数排序
    order = np.argsort(-np.array(predicted_scores))[:k]
    gains = np.array(true_scores)[order]

    # 计算 DCG
    dcg = np.sum(gains / np.log2(np.arange(2, len(gains) + 2)))

    # 计算理想 DCG
    ideal_gains = np.sort(true_scores)[::-1][:k]
    idcg = np.sum(ideal_gains / np.log2(np.arange(2, len(ideal_gains) + 2)))

    return dcg / idcg if idcg > 0 else 0.0

def evaluate_retrieval(predictions, dataset):
    """
    评估检索性能

    参数:
        predictions: 每个问题的检索答案 ID 列表
        dataset: 数据集样本
    """
    mrr_scores = []
    ndcg_scores = []

    for pred_ids, example in zip(predictions, dataset):
        answers = example['answers']

        # MRR：找到任意相关答案
        relevant_ids = set(answers['a_id'])
        mrr = compute_mrr(pred_ids, relevant_ids)
        mrr_scores.append(mrr)

        # NDCG：按 score 排序
        if len(answers['score']) > 1:
            pred_indices = [answers['a_id'].index(pid) for pid in pred_ids if pid in answers['a_id']]
            pred_scores = [1.0 / (i + 1) for i in range(len(pred_indices))]  # 按检索顺序
            true_scores = [answers['score'][i] for i in pred_indices]
            ndcg = compute_ndcg(pred_scores, true_scores)
            ndcg_scores.append(ndcg)

    return {
        'mrr': sum(mrr_scores) / len(mrr_scores),
        'ndcg': sum(ndcg_scores) / len(ndcg_scores) if ndcg_scores else 0
    }
```

### 答案质量评估

```python
from rouge_score import rouge_scorer

def evaluate_answer_quality(generated_answers, dataset):
    """
    评估生成答案质量

    参数:
        generated_answers: 生成的答案列表
        dataset: 数据集样本
    """
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    rouge_scores = {'rouge1': [], 'rouge2': [], 'rougeL': []}

    for gen, example in zip(generated_answers, dataset):
        answers = example['answers']

        # 与最高分答案比较
        if answers['score']:
            best_idx = answers['score'].index(max(answers['score']))
            reference = answers['text'][best_idx]

            scores = scorer.score(reference, gen)
            for key in rouge_scores:
                rouge_scores[key].append(scores[key].fmeasure)

    return {
        key: sum(values) / len(values) if values else 0
        for key, values in rouge_scores.items()
    }
```

## 建议方案

1. **稠密检索**：使用句向量模型进行语义检索

2. **答案重排序**：训练重排序模型基于质量分数

3. **RAG 生成**：检索相关答案后生成综合回答

4. **多答案聚合**：整合多个高分答案的信息

## 核心挑战

- 长文本答案（平均数百词）
- 答案质量参差不齐
- 需要通俗易懂的解释
- 涵盖广泛的知识领域

## 参考资料

- 原始数据: Reddit r/explainlikeimfive
- HuggingFace: https://huggingface.co/datasets/Pavithree/eli5
