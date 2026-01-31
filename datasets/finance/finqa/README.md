# FinQA 数据集

## 概述

**FinQA** 是 IBM 研究院发布的金融数值推理问答数据集，专注于在金融报表上进行**数值推理和计算**。

- **发布机构**: IBM Research
- **论文**: "FinQA: A Dataset of Numerical Reasoning over Financial Data" (EMNLP 2021)
- **HuggingFace**: `dreamerdeo/finqa`
- **许可证**: CC BY 4.0
- **语言**: 英语

## 数据集特点

- 基于**真实上市公司财报**（10-K/10-Q 报告）
- 需要对**表格和文本**进行联合推理
- 标注了完整的**推理程序**（DSL 格式），可解释性强
- 由**金融专家**标注，保证专业性

## 数据集规模

| 子集 | 样本数 | 说明 |
|------|--------|------|
| train | ~6,250 | 训练集 |
| validation | ~880 | 验证集 |
| test | ~1,150 | 测试集 |

**总计**: 约 8,280 个问答对，涵盖约 2,800 份财务报告

## 数据字段说明

| 字段名 | 类型 | 说明 |
|--------|------|------|
| `id` | string | 问题唯一标识符 |
| `question` | string | 用自然语言提出的问题 |
| `pre_text` | list | 表格前的文本段落 |
| `post_text` | list | 表格后的文本段落 |
| `table` | list | 财务表格数据（二维列表） |
| `program` | string | 推理程序（DSL 格式） |
| `answer` | string | 最终答案（通常是数值） |
| `gold_inds` | dict | 答案依据的证据索引 |

### table 字段结构

表格以二维列表形式存储：
```json
[
  ["", "2019", "2018", "2017"],
  ["Revenue", "100.0", "95.0", "90.0"],
  ["Net Income", "20.0", "18.0", "15.0"]
]
```

### program 字段（推理程序）

FinQA 使用特定的 DSL（Domain Specific Language）表示推理过程：

```
subtract(100, 95), divide(#0, 95), multiply(#1, 100)
```

常用操作符：
- `add(a, b)`: 加法
- `subtract(a, b)`: 减法
- `multiply(a, b)`: 乘法
- `divide(a, b)`: 除法
- `exp(a, b)`: 指数
- `greater(a, b)`: 比较

## 数据示例

```json
{
  "id": "example_001",
  "question": "What is the percentage change in revenue from 2018 to 2019?",
  "pre_text": ["The following table shows our revenue..."],
  "post_text": ["Revenue increased due to..."],
  "table": [
    ["Year", "2019", "2018"],
    ["Revenue (millions)", "120", "100"]
  ],
  "program": "subtract(120, 100), divide(#0, 100), multiply(#1, 100)",
  "answer": "20.0",
  "gold_inds": {
    "table_1": "Revenue (millions): 120, 100"
  }
}
```

## 评测指标

- **Execution Accuracy**: 执行推理程序后答案正确的比例
- **Program Accuracy**: 推理程序与标准程序一致的比例

## 使用方法

```python
from datasets import load_dataset

# 加载数据集
dataset = load_dataset("dreamerdeo/finqa")

# 查看训练集
print(dataset["train"][0])

# 查看表格数据
example = dataset["train"][0]
for row in example["table"]:
    print(row)
```

## 应用场景

1. **金融智能问答**: 构建能理解财报的 QA 系统
2. **数值推理研究**: 研究模型的数学推理能力
3. **表格理解**: 训练模型理解结构化表格数据
4. **可解释 AI**: 利用推理程序提供可解释的答案

## 相关数据集

| 数据集 | 特点 |
|--------|------|
| TAT-QA | 表格+文本混合 QA |
| HybridQA | 表格+文本混合推理 |
| WikiTableQuestions | Wikipedia 表格 QA |

## 参考链接

- 官网: https://finqasite.github.io/
- GitHub: https://github.com/czyssrs/FinQA
- 论文: https://aclanthology.org/2021.emnlp-main.300/
