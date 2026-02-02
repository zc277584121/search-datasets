# ELI5 问答检索

## 任务描述

ELI5 (Explain Like I'm 5) 是一个长篇问答数据集，来自 Reddit 的 r/explainlikeimfive 社区。

## 数据集信息

- **来源**: `Pavithree/eli5`
- **评测集**: 500 条
- **语言**: 英语

## 数据格式

### queries.json 字段说明

```json
{
  "task": "eli5",
  "total": 500,
  "queries": [
    {
      "id": "q_0",
      "question": "Why is the sky blue?",
      "selftext": "I've always wondered about this..."
    }
  ]
}
```

| 字段 | 类型 | 说明 |
|------|------|------|
| `id` | string | 问题唯一标识符 |
| `question` | string | 问题标题 |
| `selftext` | string | 问题的补充说明 |

## 使用流程

### 1. 加载评测数据

```python
import json

with open("queries.json", "r") as f:
    data = json.load(f)

predictions = {}
for query in data["queries"]:
    qid = query["id"]
    question = query["question"]

    # 检索相关文档
    retrieved_docs = your_retriever.search(question)
    predictions[qid] = [doc.id for doc in retrieved_docs]
```

### 2. 生成预测结果

```json
{
  "model_name": "your-model-name",
  "predictions": {
    "q_0": ["doc_physics_1", "doc_science_3", "doc_eli5_7"],
    "q_1": ["doc_biology_5", "doc_eli5_1", "doc_science_8"]
  }
}
```

**说明**:
- 每个问题返回按相关性排序的文档 ID 列表
- 建议返回 Top-10 或更多结果

### 3. 运行评估

```bash
python eval.py --submission predictions.json
```

## 评估指标

| 指标 | 说明 |
|------|------|
| **MRR** | 平均倒数排名 |
| **NDCG@10** | 归一化折损累积增益 |
| **R@K** | Recall@K |

## 输出示例

```json
{
  "task": "eli5",
  "model_name": "your-model",
  "mrr": 42.5,
  "ndcg@10": 38.7,
  "r@1": 28.3,
  "r@5": 52.1,
  "num_queries": 500
}
```

## 参考资料

- [ELI5 论文](https://arxiv.org/abs/1907.09190)
