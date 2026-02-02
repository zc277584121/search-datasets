# SQuAD 2.0 阅读理解

## 任务描述

SQuAD 2.0 是斯坦福问答数据集的第二版，包含可回答和不可回答的问题。模型需要从给定段落中抽取答案，或判断问题无法回答。

## 数据集信息

- **来源**: `rajpurkar/squad_v2`
- **评测集**: 500 条（从 validation set 采样）
- **语言**: 英语

## 数据格式

### queries.json 字段说明

```json
{
  "task": "squad2",
  "total": 500,
  "queries": [
    {
      "id": "56ddde6b9a695914005b9628",
      "context": "The Normans were the people who...",
      "question": "In what country is Normandy located?"
    }
  ]
}
```

| 字段 | 类型 | 说明 |
|------|------|------|
| `id` | string | 问题唯一标识符 |
| `context` | string | 阅读材料段落 |
| `question` | string | 问题文本 |

## 使用流程

### 1. 加载评测数据

```python
import json

with open("queries.json", "r") as f:
    data = json.load(f)

for query in data["queries"]:
    context = query["context"]
    question = query["question"]
    qid = query["id"]

    # 用你的模型预测答案
    answer = your_model.predict(context, question)
```

### 2. 生成预测结果

将预测结果写入 `predictions.json`：

```json
{
  "model_name": "your-model-name",
  "predictions": {
    "56ddde6b9a695914005b9628": "France",
    "56ddde6b9a695914005b9629": "10th and 11th centuries",
    "5ad39d53604f3c001a3fe8d1": ""
  }
}
```

**说明**:
- `predictions` 是问题 ID 到答案的映射
- 不可回答的问题应返回空字符串 `""`

### 3. 运行评估

```bash
python eval.py --submission predictions.json
```

## 评估指标

| 指标 | 说明 |
|------|------|
| **Exact Match (EM)** | 预测答案与标准答案完全匹配的比例 |
| **F1** | 预测答案与标准答案在词级别的 F1 分数 |

## 输出示例

```json
{
  "task": "squad2",
  "model_name": "your-model",
  "exact_match": 72.5,
  "f1": 81.3,
  "has_answer_em": 78.2,
  "has_answer_f1": 85.6,
  "no_answer_em": 66.8,
  "no_answer_f1": 66.8,
  "num_samples": 500
}
```

## 参考资料

- [SQuAD 2.0 论文](https://arxiv.org/abs/1806.03822)
- [官方排行榜](https://rajpurkar.github.io/SQuAD-explorer/)
