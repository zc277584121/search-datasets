# CMRC 2018 数据集

## 概述

**CMRC 2018**（Chinese Machine Reading Comprehension 2018）是哈尔滨工业大学发布的中文阅读理解数据集，是中文版的 SQuAD，也是中文 NLP 领域最重要的阅读理解基准之一。

- **发布机构**: 哈尔滨工业大学讯飞联合实验室
- **论文**: "A Span-Extraction Dataset for Chinese Machine Reading Comprehension" (EMNLP-IJCNLP 2019)
- **HuggingFace**: `hfl/cmrc2018`
- **许可证**: CC BY-SA 4.0
- **语言**: 中文

## 数据集特点

- 采用与 SQuAD 相同的**片段抽取式**问答形式
- 问题和答案都由人工标注
- 包含一个**挑战集 (challenge set)**，需要跨句推理
- 文章来源于中文 Wikipedia

## 数据集规模

| 子集 | 样本数 | 说明 |
|------|--------|------|
| train | ~10,000 | 训练集 |
| validation | ~3,200 | 验证集 |
| test | ~4,800 | 测试集（答案不公开） |
| challenge | ~500 | 挑战集，需要复杂推理 |

## 数据字段说明

| 字段名 | 类型 | 说明 |
|--------|------|------|
| `id` | string | 问题的唯一标识符 |
| `context` | string | 包含答案的段落文本 |
| `question` | string | 针对段落提出的问题 |
| `answers` | dict | 答案信息 |

### answers 字段结构

```json
{
  "text": ["答案文本1", "答案文本2", ...],
  "answer_start": [起始位置1, 起始位置2, ...]
}
```

- 每个问题可能有多个标注答案（来自不同标注者）
- `answer_start` 是答案在 context 中的字符起始位置

## 数据示例

```json
{
  "id": "TRAIN_186_QUERY_0",
  "context": "《战国无双3》（）是由光荣和ω-able开发的战国无双系列的正统第三续作...",
  "question": "《战国无双3》是由哪两个公司合作开发的？",
  "answers": {
    "text": ["光荣和ω-able"],
    "answer_start": [Mo11]
  }
}
```

## 挑战集特点

challenge 子集包含需要**多句推理**的问题：
- 答案不能直接从单句中找到
- 需要综合理解多个句子的信息
- 用于测试模型的深层理解能力

## 评测指标

- **Exact Match (EM)**: 完全匹配率
- **F1 Score**: 字符级 F1 分数

## 使用方法

```python
from datasets import load_dataset

# 加载数据集
dataset = load_dataset("hfl/cmrc2018")

# 查看各个子集
print("训练集:", len(dataset["train"]))
print("验证集:", len(dataset["validation"]))
print("测试集:", len(dataset["test"]))

# 查看示例
print(dataset["train"][0])
```

## 应用场景

1. **中文阅读理解模型训练**: 训练中文 QA 模型
2. **中文 BERT 类模型评测**: 评估预训练模型在中文理解上的效果
3. **中文检索问答系统**: 作为中文 RAG 的基准

## 与其他 CMRC 数据集的关系

| 数据集 | 年份 | 特点 |
|--------|------|------|
| CMRC 2017 | 2017 | 填空式阅读理解 |
| CMRC 2018 | 2018 | 片段抽取式（本数据集） |
| CMRC 2019 | 2019 | 句子填空式 |

## 参考链接

- 官网: https://ymcui.com/cmrc2018/
- GitHub: https://github.com/ymcui/cmrc2018
- 论文: https://aclanthology.org/D19-1600/
