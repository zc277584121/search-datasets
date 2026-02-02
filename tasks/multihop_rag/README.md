# MultiHop-RAG

## 任务描述

MultiHop-RAG 是一个多跳检索增强生成数据集，要求模型通过多步检索和推理来回答复杂问题。

## 数据集信息

- **来源**: `yixuantt/MultiHopRAG`
- **评测集**: 500 条
- **语言**: 英语

## 数据格式

### queries.json 字段说明

```json
{
  "task": "multihop_rag",
  "total": 500,
  "queries": [
    {
      "id": "0",
      "question": "Who directed the film that won Best Picture at the 95th Academy Awards?",
      "question_type": "bridge"
    }
  ]
}
```

| 字段 | 类型 | 说明 |
|------|------|------|
| `id` | string | 问题唯一标识符 |
| `question` | string | 需要多跳推理的问题 |
| `question_type` | string | 问题类型（bridge/comparison 等） |

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

    # 多跳检索并回答
    retrieved_docs = your_retriever.search(question)
    answer = your_model.answer(question, retrieved_docs)

    predictions[qid] = {
        "answer": answer,
        "retrieved_docs": [doc.id for doc in retrieved_docs]
    }
```

### 2. 生成预测结果

```json
{
  "model_name": "your-model-name",
  "predictions": {
    "0": {
      "answer": "Daniel Kwan and Daniel Scheinert",
      "retrieved_docs": ["doc_oscar_2023", "doc_eeaao_directors"]
    },
    "1": {
      "answer": "Paris",
      "retrieved_docs": ["doc_france", "doc_capital_cities"]
    }
  }
}
```

**说明**:
- `answer`: 最终答案
- `retrieved_docs`: 检索到的文档 ID 列表

### 3. 运行评估

```bash
python eval.py --submission predictions.json
```

## 评估指标

| 指标 | 说明 |
|------|------|
| **Exact Match** | 答案完全匹配的比例 |
| **F1** | 词级别的 F1 分数 |
| **Retrieval Recall** | 检索到的相关文档比例 |

## 输出示例

```json
{
  "task": "multihop_rag",
  "model_name": "your-model",
  "exact_match": 35.2,
  "f1": 52.8,
  "retrieval_recall": 68.5,
  "num_samples": 500
}
```

## 参考资料

- [MultiHop-RAG 论文](https://arxiv.org/abs/2401.15391)
