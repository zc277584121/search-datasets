# Discord 聊天检索

## 任务描述

Discord 聊天检索任务评估模型在社交聊天场景下的检索能力。数据包含真实的 Discord 服务器对话，涵盖技术讨论、日常交流等。

## 数据集信息

- **来源**: `EleutherAI/discord-code`
- **评测集**: 500 条
- **语言**: 英语

## 数据格式

### queries.json 字段说明

| 字段 | 类型 | 说明 |
|------|------|------|
| `id` | string | 消息唯一标识符 |
| `message` | string | 聊天消息内容 |

### 预测结果格式

```json
{
  "model_name": "your-model-name",
  "predictions": [
    {
      "query": "Search query",
      "retrieved": [
        {"text": "Retrieved message", "score": 0.95}
      ]
    }
  ]
}
```

## 快速开始

1. 打开 `run_demo.py`，找到 `# TODO` 注释，替换为你的模型代码
2. 运行：
   ```bash
   python run_demo.py
   ```

## 评估指标

使用 LLM-as-Judge (gpt-4o-mini) 评估：

| 指标 | 说明 |
|------|------|
| **Relevance** | 相关性评分 (1-5) |
| **Context** | 上下文质量评分 (1-5) |

## 参考资料

- [数据集页面](https://huggingface.co/datasets/EleutherAI/discord-code)
