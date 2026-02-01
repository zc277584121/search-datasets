# 任务：ChartQA 图表视觉推理

## 任务描述

构建一个多模态系统，能够回答关于**真实图表**（柱状图、折线图、饼图等）的问题，需要结合视觉理解和逻辑/数值推理能力。

## 数据集信息

| 属性 | 值 |
|------|-----|
| **名称** | ChartQA |
| **来源** | `HuggingFaceM4/ChartQA` |
| **语言** | 英语 |
| **许可证** | GPL-3.0 |
| **规模** | 训练集 ~28K，测试集 ~2.5K |

### 数据字段

| 字段 | 类型 | 描述 |
|------|------|------|
| `image` | Image | 图表图片 |
| `query` | string | 关于图表的问题 |
| `label` | string | 标准答案 |
| `source` | string | 问题来源（human/augmented） |

### 问题类型

- **Human 问题**：需要复杂的视觉 + 逻辑推理（如："最高值和最低值的差是多少？"）
- **Augmented 问题**：基于底层数据生成，主要测试数据提取能力（如："2020年的销售额是多少？"）

## 评估目标

在测试集上达到 **宽松准确率 ≥ 60%**：
- Human 问题：≥ 40%
- Augmented 问题：≥ 80%

## 评估方法

### 宽松准确率

对于数值答案，允许 5% 的误差；对于文本答案，要求完全匹配（不区分大小写）。

```python
def is_number(s):
    try:
        float(s.replace(',', '').replace('%', ''))
        return True
    except:
        return False

def parse_number(s):
    return float(s.replace(',', '').replace('%', ''))

def is_correct(prediction, ground_truth):
    """
    宽松准确率：数值答案允许 5% 误差
    """
    pred = prediction.strip()
    gold = ground_truth.strip()

    if is_number(pred) and is_number(gold):
        pred_val = parse_number(pred)
        gold_val = parse_number(gold)
        if gold_val == 0:
            return pred_val == 0
        return abs(pred_val - gold_val) <= 0.05 * abs(gold_val)
    else:
        return pred.lower() == gold.lower()

def evaluate(predictions, dataset):
    """
    predictions: 预测答案字符串列表
    dataset: 包含 'label' 和 'source' 字段的样本列表
    """
    correct = {'human': 0, 'augmented': 0}
    total = {'human': 0, 'augmented': 0}

    for pred, example in zip(predictions, dataset):
        source = example['source']
        total[source] += 1
        if is_correct(pred, example['label']):
            correct[source] += 1

    return {
        'human_accuracy': 100.0 * correct['human'] / total['human'],
        'augmented_accuracy': 100.0 * correct['augmented'] / total['augmented'],
        'overall_accuracy': 100.0 * sum(correct.values()) / sum(total.values())
    }
```

## 建议方案

1. **视觉语言模型**：使用 GPT-4V、Claude Vision 或开源 VLM（LLaVA、InternVL）

2. **图表转表格 + LLM**：将图表转换为结构化表格数据，再用 LLM 进行推理

3. **OCR + 布局分析**：从图表中提取文本和布局，然后进行推理

## 核心挑战

- 准确读取坐标轴上的数值
- 理解图表图例和颜色
- 执行多步计算
- 处理各种图表样式和质量

## 参考资料

- 论文: [ChartQA: A Benchmark for Question Answering about Charts](https://aclanthology.org/2022.findings-acl.177/)
- GitHub: https://github.com/vis-nlp/ChartQA
