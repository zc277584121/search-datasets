# CMRC 2018 中文阅读理解

## 任务描述

CMRC 2018 是中文机器阅读理解数据集，由哈工大讯飞联合实验室发布。任务要求从给定的中文段落中抽取答案来回答问题。

## 数据集信息

- **来源**: `hfl/cmrc2018`
- **评测集**: 500 条（从 validation set 采样）
- **语言**: 中文

## 数据格式

### queries.json 字段说明

```json
{
  "task": "cmrc2018",
  "total": 500,
  "queries": [
    {
      "id": "DEV_0_QUERY_0",
      "context": "《战国无双3》是由光荣和ω-force开发的...",
      "question": "《战国无双3》是由哪两个公司合作开发的？"
    }
  ]
}
```

| 字段 | 类型 | 说明 |
|------|------|------|
| `id` | string | 问题唯一标识符 |
| `context` | string | 阅读材料段落（中文） |
| `question` | string | 问题文本（中文） |

## 使用流程

### 1. 加载评测数据

```python
import json

with open("queries.json", "r", encoding="utf-8") as f:
    data = json.load(f)

for query in data["queries"]:
    context = query["context"]
    question = query["question"]
    qid = query["id"]

    # 用你的模型预测答案
    answer = your_model.predict(context, question)
```

### 2. 生成预测结果

```json
{
  "model_name": "your-model-name",
  "predictions": {
    "DEV_0_QUERY_0": "光荣和ω-force",
    "DEV_0_QUERY_1": "任天堂游戏谜之村雨城"
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
| **Exact Match (EM)** | 预测答案与标准答案完全匹配的比例 |
| **F1** | 预测答案与标准答案在**字符级别**的 F1 分数 |

注意：中文评估在字符级别进行，而非词级别。

## 输出示例

```json
{
  "task": "cmrc2018",
  "model_name": "your-model",
  "exact_match": 65.3,
  "f1": 84.7,
  "num_samples": 500
}
```

## 参考资料

- [CMRC 2018 论文](https://arxiv.org/abs/1810.07366)
- [哈工大讯飞联合实验室](https://github.com/ymcui/cmrc2018)
