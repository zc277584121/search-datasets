# Enron 邮件垃圾分类

## 任务描述

Enron 邮件数据集包含安然公司员工的真实电子邮件，任务是判断邮件是否为垃圾邮件。

## 数据集信息

- **来源**: `SetFit/enron_spam`
- **评测集**: 500 条
- **语言**: 英语
- **标签**: spam（垃圾邮件）/ ham（正常邮件）

## 数据格式

### queries.json 字段说明

```json
{
  "task": "enron",
  "total": 500,
  "queries": [
    {
      "id": "0",
      "text": "Subject: Meeting Tomorrow\n\nHi team, just a reminder about our meeting tomorrow at 2pm..."
    }
  ]
}
```

| 字段 | 类型 | 说明 |
|------|------|------|
| `id` | string | 邮件唯一标识符 |
| `text` | string | 邮件完整文本 |

## 使用流程

### 1. 加载评测数据

```python
import json

with open("queries.json", "r") as f:
    data = json.load(f)

predictions = {}
for query in data["queries"]:
    qid = query["id"]
    email_text = query["text"]

    # 用你的模型分类
    label = your_model.classify(email_text)  # "spam" or "ham"
    predictions[qid] = label
```

### 2. 生成预测结果

```json
{
  "model_name": "your-model-name",
  "predictions": {
    "0": "ham",
    "1": "spam",
    "2": "ham"
  }
}
```

**说明**:
- 预测值为 `"spam"` 或 `"ham"`

### 3. 运行评估

```bash
python eval.py --submission predictions.json
```

## 评估指标

| 指标 | 说明 |
|------|------|
| **F1** | 垃圾邮件分类的 F1 分数 |
| **Precision** | 精确率 |
| **Recall** | 召回率 |
| **Accuracy** | 准确率 |

## 输出示例

```json
{
  "task": "enron",
  "model_name": "your-model",
  "f1": 92.5,
  "precision": 91.2,
  "recall": 93.8,
  "accuracy": 94.1,
  "num_samples": 500
}
```

## 参考资料

- [Enron Email Dataset](https://www.cs.cmu.edu/~enron/)
