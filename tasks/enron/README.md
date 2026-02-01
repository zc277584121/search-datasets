# Enron 邮件搜索

## 任务描述

Enron 邮件数据集包含安然公司员工的真实电子邮件，可用于邮件检索和垃圾邮件检测任务。本任务关注邮件检索能力评估。

## 数据集

- **来源**: `SetFit/enron_spam`
- **规模**: ~30,000 封邮件
- **语言**: 英语
- **特点**: 真实商业邮件，包含正常邮件和垃圾邮件

## 任务目标

给定用户查询，从邮件库中检索最相关的邮件。任务包括：
1. 基于内容的邮件检索
2. 垃圾邮件过滤（可选子任务）

## 评估指标

### 检索任务
| 指标 | 说明 |
|------|------|
| **LLM-as-Judge** | 使用大语言模型评估检索结果的相关性 |
| **Relevance Score** | 相关性评分 (1-5) |

### 垃圾邮件检测（可选）
| 指标 | 说明 |
|------|------|
| **F1** | 垃圾邮件分类的 F1 分数 |
| **Precision** | 精确率 |
| **Recall** | 召回率 |

## 提交格式

### 检索任务
```json
{
  "model_name": "your-model-name",
  "task_type": "retrieval",
  "predictions": [
    {
      "query": "meeting schedule next week",
      "retrieved": [
        {"email_id": "email_123", "subject": "...", "body": "..."},
        {"email_id": "email_456", "subject": "...", "body": "..."}
      ]
    }
  ]
}
```

### 垃圾邮件检测（可选）
```json
{
  "model_name": "your-model-name",
  "task_type": "spam_detection",
  "predictions": {
    "email_id_1": "spam",
    "email_id_2": "ham",
    "email_id_3": "spam"
  }
}
```

## 运行评估

```bash
# 检索任务
python eval.py --submission predictions.json --task-type retrieval --api-key YOUR_API_KEY

# 垃圾邮件检测
python eval.py --submission predictions.json --task-type spam_detection
```

## 输出示例

### 检索任务
```json
{
  "task": "enron",
  "task_type": "retrieval",
  "avg_relevance": 3.7,
  "high_relevance_ratio": 0.55,
  "num_queries": 100,
  "timestamp": "2024-01-30T12:00:00"
}
```

### 垃圾邮件检测
```json
{
  "task": "enron",
  "task_type": "spam_detection",
  "f1": 92.5,
  "precision": 91.2,
  "recall": 93.8,
  "accuracy": 94.1,
  "num_samples": 5000,
  "timestamp": "2024-01-30T12:00:00"
}
```

## 数据特点

- **商业邮件**: 会议安排、项目讨论、合同往来
- **个人邮件**: 员工之间的私人通信
- **垃圾邮件**: 广告、钓鱼等垃圾邮件

## 参考资料

- [Enron Email Dataset](https://www.cs.cmu.edu/~enron/)
- [数据集页面](https://huggingface.co/datasets/SetFit/enron_spam)
