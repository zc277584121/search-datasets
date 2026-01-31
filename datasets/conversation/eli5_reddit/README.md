# ELI5 Reddit

## 数据集概述

ELI5 (Explain Like I'm 5) 是来自 Reddit 问答社区的数据集。r/explainlikeimfive 是 Reddit 上著名的问答板块，用户以简单易懂的方式解释复杂概念。该数据集非常适合长文本问答和开放域检索任务。

| 属性 | 值 |
|------|-----|
| **来源** | HuggingFace: `Pavithree/eli5` |
| **样本数** | 229,167 |
| **格式** | Parquet |
| **语言** | 英语 |
| **任务类型** | 长文本问答、开放域检索、知识蒸馏 |

## 数据集特点

- **通俗易懂**: 答案以简单方式解释复杂概念
- **社区投票**: 包含答案的投票分数，可用于质量筛选
- **多答案**: 每个问题通常有多个回答
- **丰富领域**: 涵盖科学、历史、技术、日常生活等各领域

## 数据分割

| 分割 | 样本数 | 文件 |
|------|--------|------|
| train | 216,147 | `train.parquet` |
| validation | 3,020 | `validation.parquet` |
| test | 10,000 | `test.parquet` |

## 字段说明

| 字段名 | 类型 | 描述 |
|--------|------|------|
| `q_id` | string | 问题唯一标识符 |
| `title` | string | 问题标题 |
| `selftext` | string | 问题补充说明（正文） |
| `document` | string | 相关文档（若有） |
| `subreddit` | string | 来源子版块名称 |
| `url` | string | 原帖链接 |
| `answers` | dict | 答案信息（包含 a_id, score, text 列表） |
| `title_urls` | list | 标题中的链接 |
| `selftext_urls` | list | 正文中的链接 |
| `answers_urls` | list | 答案中的链接 |

### answers 字段结构

| 子字段 | 类型 | 描述 |
|--------|------|------|
| `a_id` | list[string] | 答案ID列表 |
| `score` | list[int] | 答案投票分数列表 |
| `text` | list[string] | 答案文本列表 |

## 使用方法

```python
import pandas as pd

# 加载数据集
train_df = pd.read_parquet("train.parquet")
test_df = pd.read_parquet("test.parquet")

print(f"训练集: {len(train_df)}")
print(f"测试集: {len(test_df)}")

# 查看一个问答样本
sample = train_df.iloc[0]
print(f"问题: {sample['title']}")
print(f"答案数: {len(sample['answers']['text'])}")

# 获取最高分答案
answers = sample['answers']
if answers['score']:
    best_idx = answers['score'].index(max(answers['score']))
    print(f"最佳答案 (score={answers['score'][best_idx]}):")
    print(answers['text'][best_idx][:500])

# 筛选有高质量答案的问题
def has_good_answer(row, min_score=100):
    scores = row['answers']['score']
    return any(s >= min_score for s in scores) if scores else False

good_questions = train_df[train_df.apply(has_good_answer, axis=1)]
print(f"有高质量答案的问题数: {len(good_questions)}")
```

## 适用场景

1. **开放域问答**: 训练能解释复杂概念的问答系统
2. **长文本检索**: 基于问题检索相关答案
3. **答案质量排序**: 利用投票分数训练答案排序模型
4. **知识蒸馏**: 利用通俗解释训练更易理解的模型
5. **文本摘要**: 将长答案简化为更简洁的解释

## 数据示例

```json
{
  "title": "ELI5: Why does ice float on water?",
  "selftext": "I know most solids sink in their liquid form, but ice floats. Why?",
  "answers": {
    "a_id": ["abc123", "def456"],
    "score": [1250, 340],
    "text": [
      "Water is weird! Most things get smaller when they freeze, but water actually expands...",
      "It's because of hydrogen bonds between water molecules..."
    ]
  }
}
```

## 评测指标

- **检索任务**: R@1, R@5, R@10, MRR
- **问答任务**: ROUGE, BLEU, BERTScore
- **排序任务**: NDCG, MAP

## 注意事项

- 答案质量参差不齐，建议使用 score 字段筛选高质量答案
- 部分问题可能涉及主观或争议性话题
- 链接字段可能包含失效的 URL

## 许可证

请参考原始数据集的许可证说明：[HuggingFace 页面](https://huggingface.co/datasets/Pavithree/eli5)

## 更新日期

2026-01-30
