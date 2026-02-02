# WildChat 对话检索

## 任务描述

WildChat 是一个真实的人机对话数据集，收集自 ChatGPT 的实际使用记录。任务要求根据用户查询检索最相关的对话片段。

## 数据集信息

- **来源**: `allenai/WildChat`
- **评测集**: 500 条
- **语言**: 多语言（以英语为主）

## 数据格式

### queries.json 字段说明

```json
{
  "task": "wildchat",
  "total": 500,
  "queries": [
    {
      "id": "0",
      "conversation_id": "conv_abc123",
      "first_user_message": "How do I implement binary search in Python?",
      "language": "en"
    }
  ]
}
```

| 字段 | 类型 | 说明 |
|------|------|------|
| `id` | string | 查询唯一标识符 |
| `conversation_id` | string | 原始对话 ID |
| `first_user_message` | string | 用户的第一条消息 |
| `language` | string | 对话语言 |

## 使用流程

### 1. 加载评测数据

```python
import json

with open("queries.json", "r") as f:
    data = json.load(f)

predictions = []
for query in data["queries"]:
    qid = query["id"]
    user_message = query["first_user_message"]

    # 检索相关对话
    retrieved = your_retriever.search(user_message)

    predictions.append({
        "query": user_message,
        "retrieved": [
            {"conversation_id": r.conv_id, "text": r.text}
            for r in retrieved
        ]
    })
```

### 2. 生成预测结果

```json
{
  "model_name": "your-model-name",
  "predictions": [
    {
      "query": "How do I implement binary search?",
      "retrieved": [
        {
          "conversation_id": "conv_123",
          "text": "User: How to write binary search?\nAssistant: Here's a simple implementation..."
        }
      ]
    }
  ]
}
```

### 3. 运行评估

```bash
python eval.py --submission predictions.json --api-key YOUR_OPENAI_KEY
```

## 评估指标

使用 LLM-as-Judge 评估：

| 指标 | 说明 |
|------|------|
| **Relevance** | 相关性评分 (1-5) |
| **Coverage** | 查询意图覆盖度 (1-5) |

## 输出示例

```json
{
  "task": "wildchat",
  "model_name": "your-model",
  "avg_relevance": 3.8,
  "avg_coverage": 3.5,
  "high_relevance_ratio": 0.65,
  "num_queries": 500
}
```

## 参考资料

- [WildChat 论文](https://arxiv.org/abs/2405.01470)
