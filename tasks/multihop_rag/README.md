# MultiHop-RAG

## 任务描述

MultiHop-RAG 是一个多跳检索增强生成数据集，要求模型通过多步检索和推理来回答复杂问题。问题需要整合来自多个文档的信息才能回答。

## 数据集

- **来源**: `yixuantt/MultiHopRAG`
- **规模**: ~2,500 问答对
- **语言**: 英语
- **特点**: 需要 2-4 跳推理

## 任务目标

给定问题和文档库：
1. 检索相关文档（可能需要多轮检索）
2. 整合多个文档的信息
3. 进行推理并生成答案

## 评估指标

| 指标 | 说明 |
|------|------|
| **Recall** | 检索到的相关文档比例 |
| **F1** | 答案与标准答案的词级别 F1 |
| **EM** | 答案完全匹配的比例 |

## 提交格式

```json
{
  "model_name": "your-model-name",
  "predictions": {
    "question_id_1": {
      "answer": "The answer text",
      "retrieved_docs": ["doc_id_1", "doc_id_3", "doc_id_7"]
    },
    "question_id_2": {
      "answer": "Another answer",
      "retrieved_docs": ["doc_id_2", "doc_id_5"]
    }
  }
}
```

**说明**:
- `answer`: 最终答案文本
- `retrieved_docs`: 检索到的文档 ID 列表

## 运行评估

```bash
python eval.py --submission predictions.json
```

## 输出示例

```json
{
  "task": "multihop_rag",
  "exact_match": 35.2,
  "f1": 52.8,
  "retrieval_recall": 68.5,
  "retrieval_precision": 45.2,
  "avg_hops": 2.3,
  "num_samples": 609,
  "timestamp": "2024-01-30T12:00:00"
}
```

## 多跳推理示例

**问题**: Who is the director of the movie that won the Best Picture at the 95th Academy Awards?

**推理过程**:
1. 检索: 95th Academy Awards Best Picture → "Everything Everywhere All at Once"
2. 检索: Director of "Everything Everywhere All at Once" → Daniel Kwan, Daniel Scheinert

**答案**: Daniel Kwan and Daniel Scheinert (The Daniels)

## 参考资料

- [MultiHop-RAG 论文](https://arxiv.org/abs/2401.15391)
- [数据集页面](https://huggingface.co/datasets/yixuantt/MultiHopRAG)
