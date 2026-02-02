# WildChat 对话检索

## 任务描述

WildChat 对话检索任务评估模型在多轮对话场景下的检索能力。数据来自真实的人机对话，涵盖各种话题和语言风格。

## 数据集信息

- **来源**: `allenai/WildChat-1M`
- **评测集**: 500 条
- **语言**: 多语言

## 数据格式

### queries.json 字段说明

| 字段 | 类型 | 说明 |
|------|------|------|
| `id` | string | 对话唯一标识符 |
| `conversation_id` | string | 原始对话 ID |
| `first_user_message` | string | 用户首条消息 |
| `language` | string | 对话语言 |

### 预测结果格式

```json
{
  "model_name": "your-model-name",
  "predictions": [
    {
      "query": "User query",
      "retrieved": [
        {"text": "Retrieved conversation", "score": 0.95}
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
| **Coverage** | 意图覆盖程度 (1-5) |

## 数据来源

`queries.json` 从 HuggingFace `allenai/WildChat-1M` 数据集采样 500 条生成。

原始数据集格式：
```python
{
    "conversation_id": "对话ID",
    "conversation": [
        {"role": "user", "content": "用户消息"},
        {"role": "assistant", "content": "助手回复"}
    ],
    "language": "English",
    "model": "gpt-4"
}
```

## 参考资料

- [数据集页面](https://huggingface.co/datasets/allenai/WildChat-1M)
