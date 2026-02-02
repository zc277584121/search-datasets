# CUAD 合同审查

## 任务描述

CUAD (Contract Understanding Atticus Dataset) 是一个法律合同理解数据集，要求模型从合同文本中识别和抽取关键条款。

## 数据集信息

- **来源**: `theatticusproject/cuad-qa`
- **评测集**: 500 条
- **语言**: 英语
- **领域**: 法律合同

## 数据格式

### queries.json 字段说明

```json
{
  "task": "cuad",
  "total": 500,
  "queries": [
    {
      "id": "cuad_0",
      "question": "Identify the Termination for Convenience clause in this contract.",
      "context": "This Agreement shall be governed by the laws of Delaware..."
    }
  ]
}
```

| 字段 | 类型 | 说明 |
|------|------|------|
| `id` | string | 问题唯一标识符 |
| `question` | string | 询问特定条款类型的问题 |
| `context` | string | 合同文本片段 |

## 条款类型

常见条款类型包括：
- Termination for Convenience
- Non-Compete
- Exclusivity
- Limitation of Liability
- Governing Law
- Confidentiality
- Indemnification

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
    context = query["context"]

    # 用你的模型抽取条款
    answer, confidence = your_model.extract_clause(question, context)
    predictions[qid] = {
        "answer": answer,
        "confidence": confidence
    }
```

### 2. 生成预测结果

```json
{
  "model_name": "your-model-name",
  "predictions": {
    "cuad_0": {
      "answer": "This Agreement shall be governed by the laws of Delaware",
      "confidence": 0.95
    },
    "cuad_1": {
      "answer": "",
      "confidence": 0.1
    }
  }
}
```

**说明**:
- `answer`: 抽取的条款文本，空字符串表示不存在该条款
- `confidence`: 预测置信度 (0-1)

### 3. 运行评估

```bash
python eval.py --submission predictions.json
```

## 评估指标

| 指标 | 说明 |
|------|------|
| **AUPR** | 精确率-召回率曲线下面积 |
| **F1** | 条款级别的 F1 分数 |
| **Precision** | 精确率 |
| **Recall** | 召回率 |

## 输出示例

```json
{
  "task": "cuad",
  "model_name": "your-model",
  "aupr": 42.5,
  "f1": 65.3,
  "precision": 68.2,
  "recall": 62.7,
  "num_samples": 500
}
```

## 参考资料

- [CUAD 论文](https://arxiv.org/abs/2103.06268)
- [Atticus Project](https://www.atticusprojectai.org/)
