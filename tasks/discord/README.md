# Discord 聊天检索

## 任务描述

Discord 聊天检索任务评估模型在非正式对话场景下的检索能力。数据包含大量网络用语、表情符号和口语化表达。

## 数据集信息

- **来源**: `breadlicker45/discord-chat`
- **评测集**: 500 条
- **语言**: 英语

## 数据格式

### queries.json 字段说明

```json
{
  "task": "discord",
  "total": 500,
  "queries": [
    {
      "id": "0",
      "message": "anyone know a good Python tutorial? been trying to learn but most are boring lol"
    }
  ]
}
```

| 字段 | 类型 | 说明 |
|------|------|------|
| `id` | string | 消息唯一标识符 |
| `message` | string | Discord 消息文本 |

## 使用流程

### 1. 加载评测数据

```python
import json

with open("queries.json", "r") as f:
    data = json.load(f)

# 对每条消息，检索相关的回复或上下文
predictions = []
for query in data["queries"]:
    qid = query["id"]
    message = query["message"]

    # 检索相关消息
    retrieved = your_retriever.search(message)
    predictions.append({
        "query": message,
        "retrieved": [{"message_id": r.id, "text": r.text} for r in retrieved]
    })
```

### 2. 生成预测结果

```json
{
  "model_name": "your-model-name",
  "predictions": [
    {
      "query": "anyone know a good Python tutorial?",
      "retrieved": [
        {"message_id": "msg_123", "text": "check out Corey Schafer on youtube"},
        {"message_id": "msg_124", "text": "automate the boring stuff is free online"}
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
| **Context** | 上下文完整性评分 (1-5) |

## 输出示例

```json
{
  "task": "discord",
  "model_name": "your-model",
  "avg_relevance": 3.6,
  "avg_context": 3.2,
  "high_relevance_ratio": 0.58,
  "num_queries": 500
}
```

## 参考资料

- [数据集页面](https://huggingface.co/datasets/breadlicker45/discord-chat)
