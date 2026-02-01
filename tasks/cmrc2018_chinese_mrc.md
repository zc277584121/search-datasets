# 任务：CMRC 2018 中文阅读理解

## 任务描述

构建一个中文阅读理解系统，能够根据问题从段落中抽取答案片段。这是 SQuAD 的中文版本。

## 数据集信息

| 属性 | 值 |
|------|-----|
| **名称** | CMRC 2018 |
| **来源** | `hfl/cmrc2018` |
| **语言** | 中文 |
| **许可证** | CC BY-SA 4.0 |
| **规模** | 训练集 ~10K，验证集 ~3.2K |

### 数据字段

| 字段 | 类型 | 描述 |
|------|------|------|
| `id` | string | 问题唯一标识符 |
| `context` | string | 包含答案的中文段落 |
| `question` | string | 关于段落的问题 |
| `answers` | dict | 答案文本和起始位置 |

### 挑战集

包含一个特殊的**挑战集**（约 500 样本），需要跨句推理。

## 评估目标

达到：
- 验证集 **EM（精确匹配）≥ 65%**
- 验证集 **F1 分数 ≥ 80%**
- 挑战集 **EM ≥ 40%**

## 评估方法

### 字符级指标

对于中文，使用字符级分词：

```python
import re
from collections import Counter

def normalize_chinese_answer(s):
    """标准化中文文本以便比较"""
    # 去除标点和空白
    s = re.sub(r'[^\u4e00-\u9fa5a-zA-Z0-9]', '', s)
    return s.lower()

def chinese_tokenize(s):
    """将中文文本分词为字符"""
    normalized = normalize_chinese_answer(s)
    # 拆分为单个字符
    return list(normalized)

def compute_f1_chinese(prediction, ground_truth):
    """计算中文的字符级 F1"""
    pred_chars = chinese_tokenize(prediction)
    gold_chars = chinese_tokenize(ground_truth)

    common = Counter(pred_chars) & Counter(gold_chars)
    num_same = sum(common.values())

    if num_same == 0:
        return 0

    precision = num_same / len(pred_chars) if pred_chars else 0
    recall = num_same / len(gold_chars) if gold_chars else 0

    if precision + recall == 0:
        return 0

    return 2 * precision * recall / (precision + recall)

def compute_exact_chinese(prediction, ground_truth):
    """检查中文精确匹配"""
    return int(normalize_chinese_answer(prediction) == normalize_chinese_answer(ground_truth))

def evaluate(predictions, dataset):
    """
    评估中文阅读理解

    参数:
        predictions: 问题 id 到预测答案的字典
        dataset: 包含 'id' 和 'answers' 的样本列表
    """
    em_scores = []
    f1_scores = []

    for example in dataset:
        qid = example['id']
        gold_answers = example['answers']['text']
        prediction = predictions.get(qid, '')

        # 取所有标准答案的最大值
        em = max(compute_exact_chinese(prediction, ga) for ga in gold_answers)
        f1 = max(compute_f1_chinese(prediction, ga) for ga in gold_answers)

        em_scores.append(em)
        f1_scores.append(f1)

    return {
        'exact_match': 100.0 * sum(em_scores) / len(em_scores),
        'f1': 100.0 * sum(f1_scores) / len(f1_scores)
    }
```

## 建议方案

1. **中文 BERT**：使用 MacBERT、RoBERTa-wwm 或 Chinese-BERT-wwm

2. **多语言模型**：XLM-RoBERTa、mBERT

3. **大语言模型**：Qwen、ChatGLM 或具有中文能力的 GPT-4

4. **检索 + 阅读器**：用于更长的上下文

## 核心挑战

- 中文分词差异
- 处理文言文与现代汉语
- 跨句推理（挑战集）
- 字符级与词级匹配

## 数据示例

```json
{
  "id": "TRAIN_186_QUERY_0",
  "context": "《战国无双3》是由光荣和ω-able开发的战国无双系列的正统第三续作...",
  "question": "《战国无双3》是由哪两个公司合作开发的？",
  "answers": {
    "text": ["光荣和ω-able"],
    "answer_start": [11]
  }
}
```

## 参考资料

- 论文: [A Span-Extraction Dataset for Chinese Machine Reading Comprehension](https://aclanthology.org/D19-1600/)
- GitHub: https://github.com/ymcui/cmrc2018
- 官网: https://ymcui.com/cmrc2018/
