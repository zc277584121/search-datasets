# 任务：SQuAD 2.0 阅读理解

## 任务描述

构建一个阅读理解系统，能够根据维基百科段落回答问题，并且能够判断问题在给定上下文中是否**无法回答**。

## 数据集信息

| 属性 | 值 |
|------|-----|
| **名称** | SQuAD 2.0 (Stanford Question Answering Dataset) |
| **来源** | `rajpurkar/squad_v2` |
| **语言** | 英语 |
| **许可证** | CC BY-SA 4.0 |
| **规模** | 训练集 ~130K，验证集 ~12K |

### 数据字段

| 字段 | 类型 | 描述 |
|------|------|------|
| `id` | string | 问题唯一标识符 |
| `context` | string | 包含答案的维基百科段落 |
| `question` | string | 关于段落的问题 |
| `answers` | dict | 答案文本和起始位置（无法回答时为空） |

### 核心挑战

SQuAD 2.0 包含 50,000+ 个**无法回答的问题**，这些问题看起来合理但无法从给定上下文中找到答案。系统必须学会在适当时候回答"我不知道"。

## 评估目标

构建的系统需达到：
- 可回答问题 **EM（精确匹配）≥ 70%**
- 整体 **F1 分数 ≥ 75%**
- 无法回答检测 **F1 ≥ 70%**

## 评估方法

### 指标说明

1. **精确匹配 (EM)**：预测答案与标准答案完全一致的比例（标准化后）

2. **F1 分数**：预测与标准答案之间的词级 F1

3. **无法回答检测**：对于答案为空的问题，系统应返回空字符串

### 评估脚本

```python
import re
import string
from collections import Counter

def normalize_answer(s):
    """小写化，去除标点、冠词和多余空白"""
    def remove_articles(text):
        return re.sub(r'\b(a|an|the)\b', ' ', text)
    def white_space_fix(text):
        return ' '.join(text.split())
    def remove_punct(text):
        return ''.join(ch for ch in text if ch not in string.punctuation)
    def lower(text):
        return text.lower()
    return white_space_fix(remove_articles(remove_punct(lower(s))))

def compute_f1(prediction, ground_truth):
    pred_tokens = normalize_answer(prediction).split()
    gold_tokens = normalize_answer(ground_truth).split()
    common = Counter(pred_tokens) & Counter(gold_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0
    precision = num_same / len(pred_tokens)
    recall = num_same / len(gold_tokens)
    return (2 * precision * recall) / (precision + recall)

def compute_exact(prediction, ground_truth):
    return int(normalize_answer(prediction) == normalize_answer(ground_truth))

def evaluate(predictions, dataset):
    """
    predictions: 问题 id 到预测答案字符串的字典
    dataset: 包含 'id' 和 'answers' 字段的样本列表
    """
    f1_scores = []
    em_scores = []

    for example in dataset:
        qid = example['id']
        gold_answers = example['answers']['text']
        prediction = predictions.get(qid, '')

        if not gold_answers:  # 无法回答
            em = int(prediction == '')
            f1 = int(prediction == '')
        else:
            em = max(compute_exact(prediction, ga) for ga in gold_answers)
            f1 = max(compute_f1(prediction, ga) for ga in gold_answers)

        em_scores.append(em)
        f1_scores.append(f1)

    return {
        'exact_match': 100.0 * sum(em_scores) / len(em_scores),
        'f1': 100.0 * sum(f1_scores) / len(f1_scores)
    }
```

## 建议方案

1. **检索 + 阅读器**：使用向量检索找到相关段落，再用阅读器模型抽取答案

2. **端到端问答**：微调 BERT/RoBERTa 进行片段抽取，包含"无答案"选项

3. **基于 RAG**：使用检索增强生成配合大语言模型

## 提交格式

### 输入文件

数据集位于 `datasets/text/squad2/validation.parquet`

### 输出文件

在 `submissions/squad2/predictions.json` 中填写预测结果：

```json
{
  "model_name": "你的模型名称",
  "predictions": {
    "问题ID": "预测的答案",
    "56be85543aeaaa14008c9063": "in the late 1990s",
    "5a8d7b4c5542994a62a8d7b4": ""
  }
}
```

- 可回答问题：返回从 context 中抽取的答案文本
- 不可回答问题：返回空字符串 `""`

### 运行评估

```bash
python eval/run_eval.py --task squad2 --submission submissions/squad2/predictions.json
```

### 输出示例

```json
{
  "task": "squad2",
  "exact_match": 72.5,
  "f1": 81.3,
  "num_samples": 11873
}
```

## 参考资料

- 论文: [Know What You Don't Know: Unanswerable Questions for SQuAD](https://arxiv.org/abs/1806.03822)
- 排行榜: https://rajpurkar.github.io/SQuAD-explorer/
