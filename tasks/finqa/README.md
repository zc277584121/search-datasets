# FinQA 金融推理

## 任务描述

FinQA 是一个金融数值推理数据集，要求模型理解金融文档（包含表格和文本），并通过多步推理回答数值计算问题。

## 数据集

- **来源**: `dreamerdeo/finqa`
- **规模**: ~8,000 问答对
- **语言**: 英语
- **领域**: 金融报表分析

## 任务目标

给定金融文档（表格 + 文本）和问题：
1. 理解表格和文本中的数值信息
2. 识别所需的计算操作
3. 生成计算程序并执行
4. 返回最终数值答案

## 评估指标

| 指标 | 说明 |
|------|------|
| **执行准确率** | 执行结果与标准答案匹配的比例 |
| **程序准确率** | 生成程序与标准程序等价的比例 |

数值比较时允许小数点后两位的舍入误差。

## 提交格式

```json
{
  "model_name": "your-model-name",
  "predictions": {
    "question_id_1": {
      "answer": "42.5",
      "program": "divide(100, 2), add(#0, -7.5)"
    },
    "question_id_2": {
      "answer": "15.3%",
      "program": "subtract(25.8, 10.5), divide(#0, 100)"
    }
  }
}
```

**说明**:
- `answer`: 最终数值答案
- `program`: 可选，生成的计算程序

## 运行评估

```bash
python eval.py --submission predictions.json
```

## 输出示例

```json
{
  "task": "finqa",
  "execution_accuracy": 58.3,
  "program_accuracy": 52.1,
  "num_samples": 1147,
  "timestamp": "2024-01-30T12:00:00"
}
```

## 程序格式

FinQA 使用领域特定语言 (DSL) 表示计算:
- `add(a, b)`: 加法
- `subtract(a, b)`: 减法
- `multiply(a, b)`: 乘法
- `divide(a, b)`: 除法
- `#N`: 引用第 N 步的结果

示例：计算 (100 / 2) + 10
```
divide(100, 2), add(#0, 10)
```

## 参考资料

- [FinQA 论文](https://arxiv.org/abs/2109.00122)
- [官方代码](https://github.com/czyssrs/FinQA)
