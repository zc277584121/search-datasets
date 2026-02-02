# ChartQA 图表问答

## 任务描述

ChartQA 评估模型对图表的视觉理解和推理能力，包含人工标注和自动生成的问题。

## 数据集信息

- **来源**: `HuggingFaceM4/ChartQA`
- **评测集**: 500 条
- **语言**: 英语

## 数据格式

### queries.json 字段说明

| 字段 | 类型 | 说明 |
|------|------|------|
| `id` | string | 问题唯一标识符 |
| `question` | string | 关于图表的问题 |
| `image_index` | int | 图表图片索引 |

### 预测结果格式

```json
{
  "model_name": "your-model-name",
  "predictions": {
    "0": "42",
    "1": "25%"
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
| **Relaxed Accuracy** | 宽松准确率（数值允许5%误差） |
| **Strict Accuracy** | 严格准确率 |

## 数据来源

`queries.json` 从 HuggingFace `HuggingFaceM4/ChartQA` 数据集的 test split 采样 500 条生成。

原始数据集格式：
```python
{
    "question": "关于图表的问题",
    "answer": "答案",
    "image": <PIL.Image>,  # 图表图片
    "type": "human/augmented"  # 人工/自动生成
}
```

## 参考资料

- [ChartQA Paper](https://arxiv.org/abs/2203.10244)
