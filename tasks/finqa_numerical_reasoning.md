# 任务：FinQA 金融数值推理

## 任务描述

构建一个系统，能够回答关于**财务报告**的数值问题，需要对来自真实公司 10-K/10-Q 报告的表格数据和文本上下文进行联合推理。

## 数据集信息

| 属性 | 值 |
|------|-----|
| **名称** | FinQA |
| **来源** | `dreamerdeo/finqa` |
| **语言** | 英语 |
| **许可证** | CC BY 4.0 |
| **规模** | 训练集 ~6.3K，测试集 ~1.1K |

### 数据字段

| 字段 | 类型 | 描述 |
|------|------|------|
| `question` | string | 自然语言问题 |
| `pre_text` | list | 表格前的文本段落 |
| `post_text` | list | 表格后的文本段落 |
| `table` | list[list] | 财务表格（二维数组） |
| `program` | string | DSL 格式的推理程序 |
| `answer` | string | 最终数值答案 |
| `gold_inds` | dict | 使用的证据索引 |

### DSL 操作符

数据集包含标注的推理程序，使用领域特定语言：
- `add(a, b)`, `subtract(a, b)`, `multiply(a, b)`, `divide(a, b)`
- `exp(a, b)`: 指数运算
- `greater(a, b)`: 比较
- `#0`, `#1`: 引用前一步结果

示例：`subtract(120, 100), divide(#0, 100), multiply(#1, 100)` → 20%

## 评估目标

达到：
- **执行准确率 ≥ 60%**：执行推理程序后答案正确
- **程序准确率 ≥ 50%**：生成的程序与标准程序一致

## 评估方法

### 执行准确率

比较生成程序的执行结果与标准答案。

```python
import re

def execute_program(program, table, texts):
    """
    执行 FinQA DSL 程序并返回结果
    """
    operations = {
        'add': lambda a, b: a + b,
        'subtract': lambda a, b: a - b,
        'multiply': lambda a, b: a * b,
        'divide': lambda a, b: a / b if b != 0 else 0,
        'exp': lambda a, b: a ** b,
        'greater': lambda a, b: a > b
    }

    # 解析并执行程序步骤
    steps = program.split(', ')
    results = []

    for step in steps:
        match = re.match(r'(\w+)\(([^,]+),\s*([^)]+)\)', step)
        if not match:
            continue
        op, arg1, arg2 = match.groups()

        # 解析参数（可能是数字或引用如 #0）
        def resolve(arg):
            arg = arg.strip()
            if arg.startswith('#'):
                return results[int(arg[1:])]
            try:
                return float(arg.replace(',', '').replace('%', ''))
            except:
                return 0

        val1, val2 = resolve(arg1), resolve(arg2)
        result = operations.get(op, lambda a, b: 0)(val1, val2)
        results.append(result)

    return results[-1] if results else 0

def is_correct(pred_answer, gold_answer, tolerance=0.01):
    """检查预测答案是否在容差范围内与标准答案匹配"""
    try:
        pred = float(str(pred_answer).replace(',', '').replace('%', ''))
        gold = float(str(gold_answer).replace(',', '').replace('%', ''))
        if gold == 0:
            return pred == 0
        return abs(pred - gold) / abs(gold) <= tolerance
    except:
        return str(pred_answer).strip() == str(gold_answer).strip()

def evaluate(predictions, dataset):
    """
    predictions: 包含 'answer' 和可选 'program' 的字典列表
    """
    exec_correct = 0
    prog_correct = 0

    for pred, example in zip(predictions, dataset):
        if is_correct(pred['answer'], example['answer']):
            exec_correct += 1
        if pred.get('program') == example['program']:
            prog_correct += 1

    return {
        'execution_accuracy': 100.0 * exec_correct / len(dataset),
        'program_accuracy': 100.0 * prog_correct / len(dataset)
    }
```

## 建议方案

1. **程序生成**：训练模型生成 DSL 程序，然后执行

2. **思维链**：使用 LLM 配合逐步推理提示

3. **表格-文本融合**：结合表格理解和文本理解

4. **工具增强 LLM**：给 LLM 提供计算器工具

## 核心挑战

- 正确解析带有合并单元格的财务表格
- 多步数值计算
- 整合文本和表格信息
- 处理百分比计算和单位转换

## 参考资料

- 论文: [FinQA: A Dataset of Numerical Reasoning over Financial Data](https://aclanthology.org/2021.emnlp-main.300/)
- GitHub: https://github.com/czyssrs/FinQA
