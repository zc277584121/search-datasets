# SQuAD 2.0 阅读理解

## 任务描述

SQuAD 2.0 是斯坦福问答数据集的第二版，包含可回答和不可回答的问题。模型需要从给定段落中抽取答案，或判断问题无法回答。

## 数据集

- **来源**: `rajpurkar/squad_v2`
- **规模**: ~150,000 问答对
- **语言**: 英语
- **特点**: 包含约 50,000 个不可回答的问题

## 任务目标

给定一个段落和问题：
1. 如果问题可以从段落中回答，抽取答案文本
2. 如果问题无法回答，返回空字符串

## 评估指标

| 指标 | 说明 |
|------|------|
| **Exact Match (EM)** | 预测答案与标准答案完全匹配的比例 |
| **F1** | 预测答案与标准答案在词级别的 F1 分数 |

## 提交格式

```json
{
  "model_name": "your-model-name",
  "predictions": {
    "question_id_1": "extracted answer text",
    "question_id_2": "",
    "question_id_3": "another answer"
  }
}
```

**说明**:
- `predictions` 是问题 ID 到答案的映射
- 不可回答的问题应返回空字符串 `""`

## 运行评估

```bash
python eval.py --submission predictions.json
```

## 输出示例

```json
{
  "task": "squad2",
  "exact_match": 72.5,
  "f1": 81.3,
  "has_answer_em": 78.2,
  "has_answer_f1": 85.6,
  "no_answer_em": 66.8,
  "no_answer_f1": 66.8,
  "num_samples": 11873,
  "timestamp": "2024-01-30T12:00:00"
}
```

## 参考资料

- [SQuAD 2.0 论文](https://arxiv.org/abs/1806.03822)
- [官方排行榜](https://rajpurkar.github.io/SQuAD-explorer/)
