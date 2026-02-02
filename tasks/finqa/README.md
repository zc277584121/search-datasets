# FinQA 金融推理

## 任务描述

FinQA 是一个金融数值推理数据集，要求模型理解金融表格和文本，并通过多步推理回答数值计算问题。

## 数据集信息

- **来源**: `dreamerdeo/finqa`
- **评测集**: 500 条
- **语言**: 英语
- **领域**: 金融报表分析

## 数据格式

### queries.json 字段说明

```json
{
  "task": "finqa",
  "total": 500,
  "queries": [
    {
      "id": "finqa_0",
      "question": "What is the percentage change from 2018 to 2019?",
      "table": [
        ["Year", "Revenue ($M)"],
        ["2017", "5420"],
        ["2018", "4850"],
        ["2019", "6210"]
      ],
      "pre_text": ["The following table shows financial data:"],
      "post_text": ["Please calculate based on the data provided."]
    }
  ]
}
```

| 字段 | 类型 | 说明 |
|------|------|------|
| `id` | string | 问题唯一标识符 |
| `question` | string | 金融问题 |
| `table` | list[list] | 二维表格数据，第一行为表头 |
| `pre_text` | list[string] | 表格前的文本描述 |
| `post_text` | list[string] | 表格后的文本描述 |

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
    table = query["table"]
    context = query["pre_text"] + query["post_text"]

    # 用你的模型进行数值推理
    answer = your_model.financial_reasoning(question, table, context)
    predictions[qid] = {"answer": str(answer)}
```

### 2. 生成预测结果

```json
{
  "model_name": "your-model-name",
  "predictions": {
    "finqa_0": {
      "answer": "28.04",
      "program": "subtract(6210, 4850), divide(#0, 4850), multiply(#1, 100)"
    },
    "finqa_1": {
      "answer": "16480"
    }
  }
}
```

**说明**:
- `answer`: 必填，最终数值答案
- `program`: 可选，计算过程（用于程序准确率评估）

### 3. 运行评估

```bash
python eval.py --submission predictions.json
```

## 评估指标

| 指标 | 说明 |
|------|------|
| **执行准确率** | 数值答案与标准答案匹配（允许小数点后两位误差） |
| **程序准确率** | 生成程序与标准程序等价（可选） |

## 输出示例

```json
{
  "task": "finqa",
  "model_name": "your-model",
  "execution_accuracy": 58.3,
  "num_samples": 500
}
```

## 参考资料

- [FinQA 论文](https://arxiv.org/abs/2109.00122)
- [官方代码](https://github.com/czyssrs/FinQA)
