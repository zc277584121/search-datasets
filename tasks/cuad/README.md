# CUAD 合同审查

## 任务描述

CUAD (Contract Understanding Atticus Dataset) 是一个法律合同理解数据集，要求模型从合同文本中识别和抽取关键条款。任务涉及 41 种不同类型的法律条款识别。

## 数据集

- **来源**: `theatticusproject/cuad-qa`
- **规模**: ~13,000 条款标注
- **语言**: 英语
- **领域**: 法律合同

## 任务目标

给定合同文本和条款类型查询：
1. 判断合同中是否存在该类型条款
2. 如果存在，抽取相关文本片段

## 评估指标

| 指标 | 说明 |
|------|------|
| **AUPR** | 精确率-召回率曲线下面积 |
| **F1** | 条款级别的 F1 分数 |
| **精确率** | 正确识别的条款比例 |
| **召回率** | 被召回的条款比例 |

## 提交格式

```json
{
  "model_name": "your-model-name",
  "predictions": {
    "question_id_1": {
      "answer": "The agreement shall terminate on December 31, 2025.",
      "confidence": 0.95
    },
    "question_id_2": {
      "answer": "",
      "confidence": 0.1
    }
  }
}
```

**说明**:
- `answer`: 抽取的条款文本，空字符串表示不存在该条款
- `confidence`: 预测置信度 (0-1)，用于计算 AUPR

## 运行评估

```bash
python eval.py --submission predictions.json
```

## 输出示例

```json
{
  "task": "cuad",
  "aupr": 42.5,
  "f1": 65.3,
  "precision": 68.2,
  "recall": 62.7,
  "num_samples": 4182,
  "timestamp": "2024-01-30T12:00:00"
}
```

## 条款类型

CUAD 包含 41 种条款类型，例如：
- Termination for Convenience
- Non-Compete
- Exclusivity
- License Grant
- Limitation of Liability
- ...

## 参考资料

- [CUAD 论文](https://arxiv.org/abs/2103.06268)
- [Atticus Project](https://www.atticusprojectai.org/)
