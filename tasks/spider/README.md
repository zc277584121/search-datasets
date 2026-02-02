# Spider Text-to-SQL

## 任务描述

Spider 是一个跨领域的 Text-to-SQL 数据集，要求模型将自然语言问题转换为 SQL 查询语句。

## 数据集信息

- **来源**: `xlangai/spider`
- **评测集**: 1034 条（完整 validation set）
- **语言**: 英语
- **数据库数**: 200+

## 数据格式

### queries.json 字段说明

```json
{
  "task": "spider",
  "total": 1034,
  "queries": [
    {
      "id": "0",
      "question": "How many singers do we have?",
      "db_id": "concert_singer"
    }
  ]
}
```

| 字段 | 类型 | 说明 |
|------|------|------|
| `id` | string | 问题唯一标识符 |
| `question` | string | 自然语言问题 |
| `db_id` | string | 对应的数据库名称 |

### 获取数据库 Schema

```python
from datasets import load_dataset

dataset = load_dataset('xlangai/spider', split='validation')

# 每条数据包含数据库 schema 信息
for item in dataset:
    db_id = item['db_id']
    # item 中包含表结构信息
```

## 使用流程

### 1. 加载评测数据

```python
import json

with open("queries.json", "r") as f:
    data = json.load(f)

predictions = {}
for query in data["queries"]:
    qid = query["id"]
    question = query["question"]
    db_id = query["db_id"]

    # 加载数据库 schema（需要从原始数据集获取）
    # 用你的模型生成 SQL
    sql = your_model.text_to_sql(question, db_schema)
    predictions[qid] = sql
```

### 2. 生成预测结果

```json
{
  "model_name": "your-model-name",
  "predictions": {
    "0": "SELECT COUNT(*) FROM singer",
    "1": "SELECT name FROM concert WHERE year = 2014"
  }
}
```

### 3. 运行评估

```bash
python eval.py --submission predictions.json

# 如需执行准确率评估，需提供数据库文件
python eval.py --submission predictions.json --db-dir /path/to/spider/databases
```

## 评估指标

| 指标 | 说明 |
|------|------|
| **精确匹配 (EM)** | SQL 语句标准化后完全匹配 |
| **执行准确率 (EX)** | SQL 执行结果与标准答案一致 |

## 输出示例

```json
{
  "task": "spider",
  "model_name": "your-model",
  "exact_match": 65.3,
  "execution_accuracy": 72.5,
  "easy_accuracy": 85.2,
  "medium_accuracy": 71.4,
  "hard_accuracy": 58.6,
  "num_samples": 1034
}
```

## 参考资料

- [Spider 论文](https://arxiv.org/abs/1809.08887)
- [官方排行榜](https://yale-lily.github.io/spider)
