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

## 参考资料

- [CUAD Dataset](https://www.atticusprojectai.org/cuad)
