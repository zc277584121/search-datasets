# ChartQA 数据集

## 概述

**ChartQA** 是一个大规模图表问答数据集，专注于对**真实图表**进行视觉和逻辑推理。

- **发布机构**: Adobe Research & Georgia Tech
- **论文**: "ChartQA: A Benchmark for Question Answering about Charts with Visual and Logical Reasoning" (ACL Findings 2022)
- **HuggingFace**: `HuggingFaceM4/ChartQA`
- **许可证**: GPL-3.0
- **语言**: 英语

## 数据集特点

- **真实图表**: 来自网络的真实世界图表（非合成）
- **两种问题来源**:
  - **Human**: 人工编写的问题，需要复杂推理
  - **Augmented**: 自动生成的问题，基于图表数据
- **多样图表类型**: 柱状图、折线图、饼图等
- **需要视觉+逻辑推理**: 不仅要看图，还要做计算

## 数据集规模

| 子集 | Human 问题 | Augmented 问题 | 总计 |
|------|-----------|----------------|------|
| train | 7,398 | 20,901 | 28,299 |
| validation | 960 | 1,250 | 2,210 |
| test | 1,250 | 1,250 | 2,500 |

**总计**: ~32,000 个问答对，~21,000 张图表图片

## 数据字段说明

| 字段名 | 类型 | 说明 |
|--------|------|------|
| `image` | Image | 图表图片 |
| `query` | string | 关于图表的问题 |
| `label` | string | 问题的答案 |
| `source` | string | 问题来源（human/augmented） |

## 问题类型

### Human 问题特点
- 需要**视觉推理**: 识别颜色、位置、趋势
- 需要**逻辑推理**: 比较、计算、推断
- 更具挑战性

示例：
- "What is the difference between the highest and lowest values?"
- "Which category shows a declining trend?"

### Augmented 问题特点
- 基于图表底层数据自动生成
- 更直接，主要测试数据提取能力

示例：
- "What is the value of sales in 2020?"
- "How many categories are shown?"

## 数据示例

```python
{
    "image": <PIL.Image>,  # 图表图片
    "query": "What percentage of people prefer option A?",
    "label": "45",
    "source": "human"
}
```

## 评测指标

- **Relaxed Accuracy**: 允许数值答案有 5% 的误差
- **Strict Accuracy**: 完全匹配（用于非数值答案）

计算方式：
```python
def is_correct(pred, gold):
    if is_number(pred) and is_number(gold):
        return abs(float(pred) - float(gold)) <= 0.05 * abs(float(gold))
    else:
        return pred.lower() == gold.lower()
```

## 使用方法

```python
from datasets import load_dataset

# 加载数据集
dataset = load_dataset("HuggingFaceM4/ChartQA")

# 查看示例
example = dataset["train"][0]
print("Question:", example["query"])
print("Answer:", example["label"])

# 显示图片
example["image"].show()

# 按来源筛选
human_questions = dataset["train"].filter(
    lambda x: x["source"] == "human"
)
```

## 图表类型分布

| 图表类型 | 比例 |
|----------|------|
| 柱状图 (Bar) | ~40% |
| 折线图 (Line) | ~30% |
| 饼图 (Pie) | ~15% |
| 其他 | ~15% |

## 应用场景

1. **文档理解**: 自动理解报告中的图表
2. **BI 智能问答**: 对数据可视化进行问答
3. **无障碍服务**: 为视障用户描述图表内容
4. **多模态 LLM 评测**: 评估 VLM 的图表理解能力

## 相关数据集

| 数据集 | 特点 |
|--------|------|
| PlotQA | 合成图表，更大规模 |
| FigureQA | 合成图表，二元问答 |
| DVQA | 柱状图为主 |
| ChartQA Pro | ChartQA 的增强版 |

## 参考链接

- 官网: https://github.com/vis-nlp/ChartQA
- 论文: https://aclanthology.org/2022.findings-acl.177/
- HuggingFace: https://huggingface.co/datasets/HuggingFaceM4/ChartQA
