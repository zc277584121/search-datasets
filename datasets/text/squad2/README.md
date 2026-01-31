# SQuAD 2.0 数据集

## 概述

**SQuAD 2.0**（Stanford Question Answering Dataset 2.0）是斯坦福大学发布的阅读理解数据集，是NLP领域最经典的问答基准之一。

- **发布机构**: 斯坦福大学
- **论文**: "Know What You Don't Know: Unanswerable Questions for SQuAD" (ACL 2018)
- **HuggingFace**: `rajpurkar/squad_v2`
- **许可证**: CC BY-SA 4.0
- **语言**: 英语

## 数据集特点

SQuAD 2.0 在 SQuAD 1.1 的基础上增加了**不可回答的问题**：
- 包含 100,000+ 可回答问题（来自 SQuAD 1.1）
- 新增 50,000+ 不可回答问题（由众包工作者编写，看起来像可回答问题）
- 模型需要判断问题是否可以根据给定段落回答

## 数据集规模

| 子集 | 样本数 | 说明 |
|------|--------|------|
| train | ~130,000 | 训练集 |
| validation | ~12,000 | 验证集（用于评测） |

## 数据字段说明

| 字段名 | 类型 | 说明 |
|--------|------|------|
| `id` | string | 问题的唯一标识符 |
| `title` | string | Wikipedia 文章标题 |
| `context` | string | 包含答案的段落文本（来自 Wikipedia） |
| `question` | string | 针对段落提出的问题 |
| `answers` | dict | 答案信息，包含 `text` 和 `answer_start` |

### answers 字段结构

```json
{
  "text": ["answer text 1", "answer text 2", ...],
  "answer_start": [起始位置1, 起始位置2, ...]
}
```

- 对于**可回答**的问题：`text` 包含答案文本，`answer_start` 包含答案在 context 中的起始字符位置
- 对于**不可回答**的问题：`text` 和 `answer_start` 都是空列表 `[]`

## 数据示例

### 可回答问题示例

```json
{
  "id": "56be85543aeaaa14008c9063",
  "title": "Beyoncé",
  "context": "Beyoncé Giselle Knowles-Carter is an American singer...",
  "question": "When did Beyoncé start becoming popular?",
  "answers": {
    "text": ["in the late 1990s"],
    "answer_start": [269]
  }
}
```

### 不可回答问题示例

```json
{
  "id": "5a8d7b4c5542994a62a8d7b4",
  "title": "Beyoncé",
  "context": "Beyoncé Giselle Knowles-Carter is an American singer...",
  "question": "What is the name of Beyoncé's first child?",
  "answers": {
    "text": [],
    "answer_start": []
  }
}
```

## 评测指标

- **Exact Match (EM)**: 预测答案与标准答案完全匹配的比例
- **F1 Score**: 预测答案与标准答案的词级 F1 分数
- 对于不可回答问题，模型应返回空字符串

## 使用方法

```python
from datasets import load_dataset

# 加载数据集
dataset = load_dataset("rajpurkar/squad_v2")

# 查看训练集
print(dataset["train"][0])

# 筛选不可回答的问题
unanswerable = dataset["train"].filter(lambda x: len(x["answers"]["text"]) == 0)
```

## 应用场景

1. **阅读理解模型训练**: 训练抽取式问答模型
2. **检索增强问答**: 作为 RAG 系统的评测基准
3. **模型鲁棒性测试**: 测试模型对不可回答问题的处理能力

## 参考链接

- 官网: https://rajpurkar.github.io/SQuAD-explorer/
- 论文: https://arxiv.org/abs/1806.03822
- Leaderboard: https://rajpurkar.github.io/SQuAD-explorer/
