# 任务：CUAD 合同条款抽取

## 任务描述

构建一个系统，能够从**法律合同**中自动识别和抽取 41 类关键条款，辅助法律专业人士进行合同审查。

## 数据集信息

| 属性 | 值 |
|------|-----|
| **名称** | CUAD (Contract Understanding Atticus Dataset) |
| **来源** | `theatticusproject/cuad-qa` |
| **语言** | 英语 |
| **许可证** | CC BY 4.0 |
| **规模** | 510 份合同，13,000+ 标注 |

### 数据字段

| 字段 | 类型 | 描述 |
|------|------|------|
| `id` | string | 问题唯一标识符 |
| `title` | string | 合同文件名 |
| `context` | string | 合同文本段落 |
| `question` | string | 关于特定条款的问题 |
| `answers` | dict | 答案文本和起始位置 |

### 41 类合同条款

包括但不限于：
- 文档基本信息：文档名称、当事方、协议日期、生效日期、到期日期
- 终止相关：续约条款、终止通知期、便利终止
- 限制条款：竞业禁止、排他性、禁止招揽
- 知识产权：IP 转让、许可授予、源代码托管
- 责任条款：无上限责任、责任上限、违约金
- 其他：审计权、保密期限、第三方受益人等

## 评估目标

达到：
- **AUPR ≥ 50%**：精确率-召回率曲线下面积
- **召回率 ≥ 80%**：法律审查对漏检更敏感

## 评估方法

### AUPR 和 F1

```python
from sklearn.metrics import precision_recall_curve, auc, f1_score
import numpy as np

def compute_aupr(predictions, ground_truths, confidences):
    """
    计算 AUPR（精确率-召回率曲线下面积）

    参数:
        predictions: 预测的答案列表
        ground_truths: 标准答案列表
        confidences: 预测置信度列表
    """
    # 将预测转换为二元标签
    y_true = [1 if gt else 0 for gt in ground_truths]
    y_scores = confidences

    precision, recall, _ = precision_recall_curve(y_true, y_scores)
    return auc(recall, precision)

def compute_span_f1(pred_spans, gold_spans):
    """
    计算片段级 F1

    参数:
        pred_spans: 预测的答案片段列表
        gold_spans: 标准答案片段列表
    """
    if not gold_spans:
        return 1.0 if not pred_spans else 0.0
    if not pred_spans:
        return 0.0

    # 计算重叠
    pred_set = set()
    for span in pred_spans:
        pred_set.update(range(span['start'], span['end']))

    gold_set = set()
    for span in gold_spans:
        gold_set.update(range(span['start'], span['end']))

    overlap = len(pred_set & gold_set)
    precision = overlap / len(pred_set) if pred_set else 0
    recall = overlap / len(gold_set) if gold_set else 0

    if precision + recall == 0:
        return 0
    return 2 * precision * recall / (precision + recall)

def evaluate_by_category(predictions, dataset):
    """
    按条款类别评估

    参数:
        predictions: 预测结果字典
        dataset: 数据集样本列表
    """
    categories = {}

    for pred, example in zip(predictions, dataset):
        # 从问题中提取类别
        question = example['question']
        category = question.split('related to ')[-1].rstrip('.')

        if category not in categories:
            categories[category] = {'correct': 0, 'total': 0}

        categories[category]['total'] += 1
        if pred == example['answers']['text']:
            categories[category]['correct'] += 1

    results = {}
    for cat, stats in categories.items():
        results[cat] = 100.0 * stats['correct'] / stats['total'] if stats['total'] > 0 else 0

    return results
```

## 建议方案

1. **法律领域模型**：使用 Legal-BERT 或在法律文本上预训练的模型

2. **长文档处理**：合同通常很长，需要滑动窗口或层次化处理

3. **多标签分类 + 抽取**：先分类是否包含条款，再抽取具体内容

4. **RAG 方法**：检索相关段落，再进行条款抽取

## 核心挑战

- 法律语言专业性强
- 合同文档通常很长（数十页）
- 同一条款可能有多种表述方式
- 需要高召回率（不能漏检关键条款）

## 数据示例

```json
{
  "id": "CreditAgreement_0001193125-18-227764_1",
  "title": "CreditAgreement_0001193125-18-227764",
  "context": "This CREDIT AGREEMENT dated as of July 25, 2018...",
  "question": "Highlight the parts (if any) of this contract related to Agreement Date.",
  "answers": {
    "text": ["July 25, 2018"],
    "answer_start": [35]
  }
}
```

## 参考资料

- 论文: [CUAD: An Expert-Annotated NLP Dataset for Legal Contract Review](https://arxiv.org/abs/2103.06268)
- 官网: https://www.atticusprojectai.org/cuad
- GitHub: https://github.com/TheAtticusProject/cuad
