# FinQA 金融数值推理

## 任务描述

FinQA 评估模型在金融报表上的数值推理能力，需要理解表格数据并进行计算。

## 数据集信息

- **来源**: `dreamerdeo/finqa`
- **评测集**: 500 条
- **语言**: 英语

## 数据格式

### queries.json 字段说明

| 字段 | 类型 | 说明 |
|------|------|------|
| `id` | string | 问题唯一标识符 |
| `question` | string | 金融问题 |
| `table` | array | 金融数据表格 |

### 预测结果格式

```json
{
  "model_name": "your-model-name",
  "predictions": {
    "finqa_0": {"answer": "15.5%", "program": ""}
  }
}
```

## 快速开始

1. 打开 `run_demo.py`，找到 `# TODO` 注释，替换为你的模型代码
2. 运行：
   ```bash
   python run_demo.py
   ```

## 评估指标

| 指标 | 说明 |
|------|------|
| **Execution Accuracy** | 数值答案的准确率 |

## 数据来源

`queries.json` 从 HuggingFace `dreamerdeo/finqa` 数据集的 test split 采样 500 条生成。

原始数据集格式：
```python
{
    "id": "问题ID",
    "question": "金融问题",
    "table": [["Year", "Revenue"], ["2019", "100M"]],  # 表格数据
    "answer": "15.5%",  # 数值答案
    "program": "divide(100, 50)"  # 推理程序（可选）
}
```

## 参考资料

- [FinQA Paper](https://arxiv.org/abs/2109.00122)
