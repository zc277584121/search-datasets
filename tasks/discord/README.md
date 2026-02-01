# Discord 聊天检索

## 任务描述

Discord 聊天检索任务评估模型在非正式对话场景下的检索能力。数据来自 Discord 聊天频道，包含大量网络用语、表情符号和口语化表达。

## 数据集

- **来源**: `breadlicker45/discord-chat`
- **规模**: ~1,000,000 条消息
- **语言**: 英语（含大量网络用语）
- **特点**: 非正式对话、表情符号、缩写

## 任务目标

给定用户查询，从 Discord 聊天记录中检索最相关的消息或对话片段。需要处理：
1. 非正式语言和俚语
2. 表情符号和特殊字符
3. 上下文依赖的对话

## 评估指标

| 指标 | 说明 |
|------|------|
| **LLM-as-Judge** | 使用大语言模型评估检索结果的相关性 |
| **Relevance Score** | 相关性评分 (1-5) |
| **Context Score** | 上下文完整性评分 (1-5) |

## 提交格式

```json
{
  "model_name": "your-model-name",
  "predictions": [
    {
      "query": "anyone know a good Python tutorial?",
      "retrieved": [
        {"message_id": "msg_123", "text": "..."},
        {"message_id": "msg_456", "text": "..."}
      ]
    }
  ]
}
```

## 运行评估

```bash
python eval.py --submission predictions.json --api-key YOUR_API_KEY
```

## 输出示例

```json
{
  "task": "discord",
  "avg_relevance": 3.6,
  "avg_context": 3.2,
  "high_relevance_ratio": 0.58,
  "num_queries": 100,
  "timestamp": "2024-01-30T12:00:00"
}
```

## 挑战

Discord 数据的特殊挑战：
- **非标准语法**: "u" 代替 "you", "rn" 代替 "right now"
- **表情符号**: :thinking: 🤔 等
- **上下文依赖**: 回复可能依赖之前的消息
- **多主题交织**: 同一频道多个话题同时进行

## 参考资料

- [数据集页面](https://huggingface.co/datasets/breadlicker45/discord-chat)
