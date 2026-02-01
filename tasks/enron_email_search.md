# 任务：Enron 邮件搜索与分类

## 任务描述

构建一个邮件搜索系统，能够：
1. 根据关键词/语义检索相关邮件
2. 识别垃圾邮件与正常邮件

## 数据集信息

| 属性 | 值 |
|------|-----|
| **名称** | Enron Spam Email |
| **来源** | `SetFit/enron_spam` |
| **语言** | 英语 |
| **许可证** | 见原始数据集 |
| **规模** | 训练集 ~31.7K，测试集 2K |

### 数据字段

| 字段 | 类型 | 描述 |
|------|------|------|
| `message_id` | int | 邮件唯一标识符 |
| `text` | string | 邮件完整文本 |
| `label` | int | 0=正常邮件，1=垃圾邮件 |
| `label_text` | string | ham/spam |
| `subject` | string | 邮件主题 |
| `message` | string | 邮件正文 |
| `date` | timestamp | 邮件日期 |

## 评估目标

### 分类任务
- **垃圾邮件检测 F1 ≥ 95%**
- **准确率 ≥ 97%**

### 检索任务
- 使用 **LLM-as-Judge** 评估检索相关性
- **相关性评分 ≥ 4/5**

## 评估方法

### 垃圾邮件分类评估

```python
from sklearn.metrics import classification_report, accuracy_score, f1_score, roc_auc_score

def evaluate_spam_classification(predictions, ground_truth):
    """
    评估垃圾邮件分类性能

    参数:
        predictions: 预测标签列表 (0/1)
        ground_truth: 真实标签列表 (0/1)
    """
    accuracy = accuracy_score(ground_truth, predictions)
    f1 = f1_score(ground_truth, predictions)

    # 详细报告
    report = classification_report(
        ground_truth, predictions,
        target_names=['ham', 'spam'],
        output_dict=True
    )

    return {
        'accuracy': accuracy * 100,
        'f1_spam': f1 * 100,
        'precision_spam': report['spam']['precision'] * 100,
        'recall_spam': report['spam']['recall'] * 100
    }

def evaluate_with_confidence(predictions, confidences, ground_truth):
    """
    带置信度的评估（用于计算 AUC）

    参数:
        predictions: 预测标签
        confidences: 预测为垃圾邮件的概率
        ground_truth: 真实标签
    """
    auc = roc_auc_score(ground_truth, confidences)

    return {
        'auc_roc': auc * 100,
        **evaluate_spam_classification(predictions, ground_truth)
    }
```

### 邮件检索评估

```python
from openai import OpenAI

def evaluate_email_retrieval(query, retrieved_emails, client):
    """
    使用 LLM 评估邮件检索相关性

    参数:
        query: 搜索查询
        retrieved_emails: 检索到的邮件列表
        client: OpenAI 客户端
    """
    scores = []

    for email in retrieved_emails[:5]:
        prompt = f"""请评估以下邮件与搜索查询的相关性。

搜索查询：{query}

邮件主题：{email['subject']}
邮件日期：{email['date']}
邮件内容（摘要）：{email['message'][:300] if email['message'] else 'N/A'}...

评分标准（1-5分）：
1 = 完全不相关
2 = 主题略微相关
3 = 部分相关
4 = 高度相关
5 = 完全匹配查询

请只输出一个数字。"""

        response = client.chat.completions.create(
            model="gpt-4",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=5
        )

        try:
            score = int(response.choices[0].message.content.strip())
        except:
            score = 3

        scores.append(score)

    return {
        'mean_relevance': sum(scores) / len(scores),
        'scores': scores
    }
```

### 主题搜索评估

```python
def evaluate_subject_search(query, retrieved_emails):
    """
    评估基于主题的精确搜索

    参数:
        query: 搜索关键词
        retrieved_emails: 检索到的邮件
    """
    query_lower = query.lower()

    # 计算包含查询词的比例
    matches = sum(
        1 for email in retrieved_emails
        if query_lower in email['subject'].lower()
    )

    precision_at_k = {
        f'P@{k}': sum(
            1 for email in retrieved_emails[:k]
            if query_lower in email['subject'].lower()
        ) / k
        for k in [1, 5, 10]
    }

    return {
        'total_matches': matches,
        **precision_at_k
    }
```

## 建议方案

1. **垃圾邮件检测**：
   - 传统方法：TF-IDF + SVM/朴素贝叶斯
   - 深度学习：BERT 分类器
   - 轻量级：SetFit 少样本学习

2. **邮件检索**：
   - BM25 关键词检索
   - 稠密向量检索
   - 混合检索（关键词 + 语义）

3. **增强特征**：
   - 发件人/收件人信息
   - 时间信息
   - 邮件线程关系

## 核心挑战

- 邮件内容年代久远（2000-2002 年）
- 垃圾邮件特征随时间演变
- 商务邮件的专业术语
- 隐私敏感内容

## 查询示例

```
查询："meeting tomorrow"

期望检索到：
- 主题包含 "meeting" 的邮件
- 讨论会议安排的邮件
- 确认或取消会议的邮件
```

## 参考资料

- 原始数据: Enron Email Dataset
- HuggingFace: https://huggingface.co/datasets/SetFit/enron_spam
