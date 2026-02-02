# WildChat 对话检索

## 任务描述

WildChat 对话检索任务评估模型检索相似对话的能力。给定用户的第一条消息，从对话语料库中检索出主题相似的对话。

## 数据集信息

- **来源**: `allenai/WildChat` (HuggingFace)
- **评测集**: 500 条 queries
- **语料库**: 5,000 条对话
- **语言**: 多语言（以英语为主）
- **有效 queries**: 280 条（通过关键词匹配找到相关文档）

## 数据格式

### queries.json 字段说明

| 字段 | 类型 | 说明 |
|------|------|------|
| `id` | string | 查询唯一标识符 |
| `conversation_id` | string | 对话 ID |
| `first_user_message` | string | 用户第一条消息 |
| `language` | string | 对话语言 |

### corpus.json 字段说明

| 字段 | 类型 | 说明 |
|------|------|------|
| `id` | string | 文档唯一标识符 |
| `conversation_id` | string | 对话 ID |
| `text` | string | 对话完整文本 |
| `language` | string | 对话语言 |

### 预测结果格式

```json
{
  "model_name": "your-model-name",
  "predictions": {
    "0": ["corpus_id_1", "corpus_id_2", "corpus_id_3"],
    "1": ["corpus_id_5", "corpus_id_8"]
  }
}
```

## 快速开始

1. 打开 `run_demo.py`，找到 `# TODO` 注释，替换为你的模型代码
2. 运行：
   ```bash
   python run_demo.py
   ```

## 评估指标

| 指标 | 说明 |
|------|------|
| **MRR** | 平均倒数排名 |
| **MAP** | 平均精度均值 |
| **Recall@K** | Top-K 召回率 |

## 数据来源

`queries.json` 从原始 WildChat 数据集采样生成。`corpus.json` 使用前 5,000 条对话。Ground truth 通过关键词匹配生成（匹配度 >= 3 个关键词）。

## 参考资料

- [WildChat Paper](https://arxiv.org/abs/2405.01470)
- [HuggingFace Dataset](https://huggingface.co/datasets/allenai/WildChat)
