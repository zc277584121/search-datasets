# ELI5 问答检索

## 任务描述

ELI5 (Explain Like I'm 5) 是一个长篇问答数据集，来自 Reddit 的 r/explainlikeimfive 社区。问题通常需要详细解释性的答案，而非简单的事实性回答。

## 数据集

- **来源**: `Pavithree/eli5`
- **规模**: ~270,000 问答对
- **语言**: 英语
- **特点**: 答案为长篇解释性文本

## 任务目标

给定问题，从文档库中检索相关文档，并生成详细的解释性答案。

## 评估指标

| 指标 | 说明 |
|------|------|
| **MRR** | 平均倒数排名 (Mean Reciprocal Rank) |
| **NDCG** | 归一化折损累积增益 |
| **R@K** | Recall@K，相关文档在前 K 位的比例 |

## 提交格式

```json
{
  "model_name": "your-model-name",
  "predictions": {
    "question_id_1": ["doc_id_1", "doc_id_3", "doc_id_7", "doc_id_2"],
    "question_id_2": ["doc_id_5", "doc_id_1", "doc_id_8", "doc_id_3"]
  }
}
```

**说明**:
- 每个问题返回按相关性排序的文档 ID 列表
- 建议返回 Top-10 或更多结果

## 运行评估

```bash
python eval.py --submission predictions.json
```

## 输出示例

```json
{
  "task": "eli5",
  "mrr": 42.5,
  "ndcg@10": 38.7,
  "r@1": 28.3,
  "r@5": 52.1,
  "r@10": 65.8,
  "num_queries": 1507,
  "timestamp": "2024-01-30T12:00:00"
}
```

## 问题类型

ELI5 的问题通常以 "Why" 或 "How" 开头，例如：
- "Why is the sky blue?"
- "How does a computer work?"
- "Why do we dream?"

## 参考资料

- [ELI5 论文](https://arxiv.org/abs/1907.09190)
- [Reddit r/explainlikeimfive](https://www.reddit.com/r/explainlikeimfive/)
