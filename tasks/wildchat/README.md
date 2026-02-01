# WildChat 对话检索

## 任务描述

WildChat 是一个真实的人机对话数据集，收集自 ChatGPT 的实际使用记录。任务要求根据用户查询检索最相关的对话片段。

## 数据集

- **来源**: `sam-paech/wildchat_*`
- **规模**: ~1,000,000 对话
- **语言**: 多语言（以英语为主）
- **特点**: 真实用户对话，涵盖多种主题

## 任务目标

给定用户查询，从对话库中检索最相关的对话。查询可能涉及：
1. 特定话题的讨论
2. 特定类型的问答
3. 特定技能的展示（如编程、写作等）

## 评估指标

| 指标 | 说明 |
|------|------|
| **LLM-as-Judge** | 使用大语言模型评估检索结果的相关性 |
| **Relevance Score** | 相关性评分 (1-5) |
| **Coverage Score** | 查询意图覆盖度 (1-5) |

由于没有标准答案，使用 LLM 评估检索质量。

## 提交格式

```json
{
  "model_name": "your-model-name",
  "predictions": [
    {
      "query": "How to implement binary search in Python?",
      "retrieved": [
        {"conversation_id": "conv_123", "text": "..."},
        {"conversation_id": "conv_456", "text": "..."}
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
  "task": "wildchat",
  "avg_relevance": 3.8,
  "avg_coverage": 3.5,
  "high_relevance_ratio": 0.65,
  "num_queries": 100,
  "timestamp": "2024-01-30T12:00:00"
}
```

## LLM 评估标准

### 相关性评分 (1-5)
- 5: 完全相关，直接回答查询
- 4: 高度相关，主题一致
- 3: 部分相关，有一些相关信息
- 2: 边缘相关，只有少量相关内容
- 1: 不相关

### 覆盖度评分 (1-5)
- 5: 完全覆盖查询意图
- 4: 大部分覆盖
- 3: 一半覆盖
- 2: 少量覆盖
- 1: 未覆盖

## 参考资料

- [WildChat 论文](https://arxiv.org/abs/2405.01470)
- [数据集页面](https://huggingface.co/datasets/allenai/WildChat)
