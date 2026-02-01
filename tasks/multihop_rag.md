# 任务：MultiHop-RAG 多文档推理

## 任务描述

构建一个检索增强生成（RAG）系统，能够回答需要从**多个文档**中获取信息的**多跳问题**。系统必须检索 2-4 个相关文档并综合它们的信息来生成最终答案。

## 数据集信息

| 属性 | 值 |
|------|-----|
| **名称** | MultiHop-RAG |
| **来源** | `yixuantt/MultiHopRAG` |
| **语言** | 英语 |
| **许可证** | MIT |
| **规模** | 2,556 个查询，609 个文档 |

### 数据字段

**查询数据：**
| 字段 | 类型 | 描述 |
|------|------|------|
| `query_id` | string | 查询唯一标识符 |
| `query` | string | 自然语言问题 |
| `answer` | string | 标准答案 |
| `question_type` | string | 多跳推理类型 |
| `evidence_list` | list | 支撑文档 ID 列表 |

**知识库数据：**
| 字段 | 类型 | 描述 |
|------|------|------|
| `doc_id` | string | 文档标识符 |
| `title` | string | 新闻标题 |
| `content` | string | 文章内容 |
| `published_at` | string | 发布日期 |
| `source` | string | 新闻来源 |

### 多跳类型

- **推理型**：链接逻辑事实（A→B→C）
- **比较型**：跨实体比较属性
- **桥接型**：通过中间实体连接
- **组合型**：整合多个独立事实

## 评估目标

达到：
- **检索 Recall@10 ≥ 80%**：所有证据文档在前 10 名
- **答案 F1 ≥ 50%**：与标准答案的词级重叠
- **答案 EM ≥ 30%**：精确匹配率

## 评估方法

### 检索评估

```python
def retrieval_recall_at_k(retrieved_docs, evidence_list, k=10):
    """
    检查所有证据文档是否在 top-k 检索结果中
    """
    top_k = set(retrieved_docs[:k])
    evidence = set(evidence_list)
    found = len(evidence & top_k)
    return found / len(evidence) if evidence else 1.0

def evaluate_retrieval(predictions, dataset):
    """
    predictions: 检索到的 doc_id 列表
    """
    recall_scores = {5: [], 10: [], 20: []}

    for retrieved, example in zip(predictions, dataset):
        evidence = example['evidence_list']
        for k in recall_scores:
            recall_scores[k].append(retrieval_recall_at_k(retrieved, evidence, k))

    return {
        f'recall@{k}': 100.0 * sum(scores) / len(scores)
        for k, scores in recall_scores.items()
    }
```

### 答案评估

```python
import re
from collections import Counter

def normalize_answer(s):
    """小写化并去除标点"""
    s = s.lower()
    s = re.sub(r'[^\w\s]', '', s)
    return ' '.join(s.split())

def compute_f1(prediction, ground_truth):
    pred_tokens = normalize_answer(prediction).split()
    gold_tokens = normalize_answer(ground_truth).split()
    common = Counter(pred_tokens) & Counter(gold_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0
    precision = num_same / len(pred_tokens) if pred_tokens else 0
    recall = num_same / len(gold_tokens) if gold_tokens else 0
    if precision + recall == 0:
        return 0
    return 2 * precision * recall / (precision + recall)

def evaluate_answer(predictions, dataset):
    """
    predictions: 答案字符串列表
    """
    em_scores = []
    f1_scores = []

    for pred, example in zip(predictions, dataset):
        gold = example['answer']
        em_scores.append(int(normalize_answer(pred) == normalize_answer(gold)))
        f1_scores.append(compute_f1(pred, gold))

    return {
        'exact_match': 100.0 * sum(em_scores) / len(em_scores),
        'f1': 100.0 * sum(f1_scores) / len(f1_scores)
    }
```

## 建议方案

1. **稠密检索 + LLM**：使用向量模型检索，然后 LLM 生成答案

2. **迭代检索**：检索 → 阅读 → 基于上下文再检索

3. **基于图的推理**：从文档构建实体图，遍历进行多跳推理

4. **重排序**：使用交叉编码器对检索结果重排序

## 核心挑战

- 证据分散在多个文档中
- 正确的推理顺序很重要
- 无关文档的噪声干扰
- 需要利用元数据（日期、来源）

## 参考资料

- 论文: [MultiHop-RAG: Benchmarking Retrieval-Augmented Generation for Multi-Hop Queries](https://arxiv.org/abs/2401.15391)
- GitHub: https://github.com/yixuantt/MultiHop-RAG
