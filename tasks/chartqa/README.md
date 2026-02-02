# ChartQA 图表推理

## 任务描述

ChartQA 是一个图表视觉问答数据集，要求模型理解图表中的视觉信息并回答相关问题。任务涵盖数据读取、比较和推理等多种能力。

## 数据集信息

- **来源**: `HuggingFaceM4/ChartQA`
- **评测集**: 500 条
- **语言**: 英语

## 数据格式

### queries.json 字段说明

```json
{
  "task": "chartqa",
  "total": 500,
  "queries": [
    {
      "id": "0",
      "question": "What is the value for 2019?",
      "image_index": 0
    }
  ]
}
```

| 字段 | 类型 | 说明 |
|------|------|------|
| `id` | string | 问题唯一标识符 |
| `question` | string | 关于图表的问题 |
| `image_index` | int | 对应图像在数据集中的索引 |

### 加载图像

```python
from datasets import load_dataset

# 加载数据集获取图像
dataset = load_dataset('HuggingFaceM4/ChartQA', split='test')

# 根据 image_index 获取图像
image = dataset[image_index]['image']
```

## 使用流程

### 1. 加载评测数据

```python
import json
from datasets import load_dataset

# 加载问题
with open("queries.json", "r") as f:
    data = json.load(f)

# 加载图像数据集
dataset = load_dataset('HuggingFaceM4/ChartQA', split='test')

predictions = {}
for query in data["queries"]:
    qid = query["id"]
    question = query["question"]
    image = dataset[query["image_index"]]['image']

    # 用你的多模态模型预测
    answer = your_model.predict(image, question)
    predictions[qid] = answer
```

### 2. 生成预测结果

```json
{
  "model_name": "your-model-name",
  "predictions": {
    "0": "42.5",
    "1": "Revenue",
    "2": "2019"
  }
}
```

### 3. 运行评估

```bash
python eval.py --submission predictions.json
```

## 评估指标

| 指标 | 说明 |
|------|------|
| **宽松准确率** | 对数值答案允许 5% 误差范围 |
| **严格准确率** | 完全匹配 |

## 输出示例

```json
{
  "task": "chartqa",
  "model_name": "your-model",
  "relaxed_accuracy": 58.3,
  "strict_accuracy": 52.1,
  "num_samples": 500
}
```

## 参考资料

- [ChartQA 论文](https://arxiv.org/abs/2203.10244)
