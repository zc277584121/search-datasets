# CUAD 合同理解

## 任务描述

CUAD (Contract Understanding Atticus Dataset) 评估模型从法律合同中提取特定条款的能力。

## 数据集信息

- **来源**: `theatticusproject/cuad-qa`
- **评测集**: 500 条
- **语言**: 英语

## 数据格式

### queries.json 字段说明

| 字段 | 类型 | 说明 |
|------|------|------|
| `id` | string | 问题唯一标识符 |
| `question` | string | 条款识别问题 |
| `context` | string | 合同文本 |

### 预测结果格式

```json
{
  "model_name": "your-model-name",
  "predictions": {
    "cuad_0": {"answer": "Extracted clause text", "confidence": 0.95}
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
| **AUPR** | 精确率-召回率曲线下面积 |
| **F1** | F1 分数 |

## 数据来源

`queries.json` 基于 CUAD 数据集格式生成 500 条合同条款识别样本。

原始数据集格式：
```python
{
    "id": "问题ID",
    "context": "合同全文...",
    "question": "Identify the Termination clause",
    "answers": {
        "text": ["条款内容"],  # 可能为空
        "answer_start": [起始位置]
    }
}
```

## 参考资料

- [CUAD Dataset](https://www.atticusprojectai.org/cuad)
