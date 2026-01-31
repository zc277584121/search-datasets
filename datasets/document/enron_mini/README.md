# Enron Spam Email

## 数据集概述

Enron Spam Email 数据集源自著名的 Enron 邮件语料库，经过整理后用于垃圾邮件检测任务。Enron 语料库是信息检索和文本挖掘领域的经典数据集，包含 Enron 公司员工的真实邮件往来。

| 属性 | 值 |
|------|-----|
| **来源** | HuggingFace: `SetFit/enron_spam` |
| **样本数** | 33,716 |
| **格式** | Parquet |
| **语言** | 英语 |
| **任务类型** | 邮件检索、垃圾邮件分类、文档搜索 |

## 数据集特点

- **真实商务邮件**: 来自 Enron 公司内部的真实邮件
- **二分类标注**: 包含正常邮件和垃圾邮件标签
- **完整邮件结构**: 包含主题、正文、日期等字段
- **历史价值**: 经典的文本分类和检索基准数据集

## 数据分割

| 分割 | 样本数 | 文件 |
|------|--------|------|
| train | 31,716 | `train.parquet` |
| test | 2,000 | `test.parquet` |

## 字段说明

| 字段名 | 类型 | 描述 |
|--------|------|------|
| `message_id` | int | 邮件唯一标识符 |
| `text` | string | 邮件完整文本（包含主题和正文） |
| `label` | int | 标签（0: 正常邮件, 1: 垃圾邮件） |
| `label_text` | string | 标签文本（ham/spam） |
| `subject` | string | 邮件主题 |
| `message` | string | 邮件正文 |
| `date` | timestamp | 邮件日期 |

## 使用方法

```python
import pandas as pd

# 加载数据集
train_df = pd.read_parquet("train.parquet")
test_df = pd.read_parquet("test.parquet")

print(f"训练集: {len(train_df)}")
print(f"测试集: {len(test_df)}")

# 查看标签分布
print("\n标签分布:")
print(train_df['label_text'].value_counts())

# 查看一封邮件样本
sample = train_df.iloc[0]
print(f"\n邮件ID: {sample['message_id']}")
print(f"主题: {sample['subject']}")
print(f"日期: {sample['date']}")
print(f"类型: {sample['label_text']}")
print(f"正文预览: {sample['message'][:300] if sample['message'] else 'N/A'}...")

# 分离正常邮件和垃圾邮件
ham_emails = train_df[train_df['label'] == 0]
spam_emails = train_df[train_df['label'] == 1]
print(f"\n正常邮件: {len(ham_emails)}")
print(f"垃圾邮件: {len(spam_emails)}")

# 按主题搜索
keyword = "meeting"
results = train_df[train_df['subject'].str.contains(keyword, case=False, na=False)]
print(f"\n包含 '{keyword}' 的邮件: {len(results)}")
```

## 适用场景

1. **邮件检索**: 基于主题或内容的邮件搜索
2. **垃圾邮件分类**: 训练二分类模型识别垃圾邮件
3. **文档搜索**: 企业文档检索系统原型
4. **文本分类基准**: 经典的文本分类评测数据集
5. **语言模型训练**: 商务邮件风格的语言模型微调

## 数据示例

```json
{
  "message_id": 12345,
  "subject": "Re: Meeting Tomorrow",
  "message": "Hi John,\n\nI confirmed the meeting room for tomorrow at 2pm...",
  "date": "2001-05-15T10:30:00",
  "label": 0,
  "label_text": "ham"
}
```

## 评测指标

- **分类任务**: Accuracy, Precision, Recall, F1-Score
- **检索任务**: R@K, MRR, MAP
- **AUC-ROC**: 用于垃圾邮件检测的综合评估

## 数据集历史

Enron 邮件语料库源自 2001 年 Enron 公司破产案期间被公开的员工邮件。该数据集后来被整理用于学术研究，成为文本分类和信息检索领域的标准基准之一。

## 注意事项

- 邮件内容来自 2000-2002 年，语言风格可能较为过时
- 部分邮件可能包含个人信息，使用时请注意隐私
- 垃圾邮件部分是后期添加的，非原始 Enron 数据

## 许可证

请参考原始数据集的许可证说明：[HuggingFace 页面](https://huggingface.co/datasets/SetFit/enron_spam)

## 更新日期

2026-01-30
