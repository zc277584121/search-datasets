# ChartQA 图表推理

## 任务描述

ChartQA 是一个图表视觉问答数据集，要求模型理解图表中的视觉信息并回答相关问题。任务涵盖数据读取、比较和推理等多种能力。

## 数据集

- **来源**: `HuggingFaceM4/ChartQA`
- **规模**: ~32,000 问答对
- **语言**: 英语
- **特点**: 包含人工标注和自动生成两类问题

## 任务目标

给定一张图表图像和问题，模型需要：
1. 理解图表的类型（柱状图、折线图、饼图等）
2. 解析图表中的数据
3. 根据问题进行推理并给出答案

## 评估指标

| 指标 | 说明 |
|------|------|
| **宽松准确率** | 对数值答案允许 5% 误差范围 |

对于数值型答案：如果预测值在标准答案的 ±5% 范围内，视为正确。
对于文本型答案：进行标准化后的精确匹配。

## 提交格式

```json
{
  "model_name": "your-model-name",
  "predictions": {
    "question_id_1": "42.5",
    "question_id_2": "Revenue",
    "question_id_3": "2019"
  }
}
```

## 运行评估

```bash
python eval.py --submission predictions.json
```

## 输出示例

```json
{
  "task": "chartqa",
  "relaxed_accuracy": 58.3,
  "strict_accuracy": 52.1,
  "human_accuracy": 45.2,
  "augmented_accuracy": 71.4,
  "num_samples": 2500,
  "timestamp": "2024-01-30T12:00:00"
}
```

## 参考资料

- [ChartQA 论文](https://arxiv.org/abs/2203.10244)
- [数据集页面](https://huggingface.co/datasets/HuggingFaceM4/ChartQA)
