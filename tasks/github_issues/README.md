# GitHub Issues 检索

## 任务描述

GitHub Issues 检索任务评估模型在技术问题追踪场景下的检索能力。数据包含真实的 GitHub issue，涵盖 bug 报告、功能请求、讨论等。

## 数据集信息

- **来源**: `lewtun/github-issues`
- **评测集**: 500 条
- **语言**: 英语

## 数据格式

### queries.json 字段说明

```json
{
  "task": "github_issues",
  "total": 500,
  "queries": [
    {
      "id": "0",
      "title": "TypeError when using async functions with decorators",
      "body": "When I apply a decorator to an async function, I get a TypeError..."
    }
  ]
}
```

| 字段 | 类型 | 说明 |
|------|------|------|
| `id` | string | Issue 唯一标识符 |
| `title` | string | Issue 标题 |
| `body` | string | Issue 正文描述 |

## 使用流程

### 1. 加载评测数据

```python
import json

with open("queries.json", "r") as f:
    data = json.load(f)

predictions = []
for query in data["queries"]:
    qid = query["id"]
    title = query["title"]
    body = query["body"]

    # 检索相关 issues
    search_query = f"{title} {body}"
    retrieved = your_retriever.search(search_query)

    predictions.append({
        "query": title,
        "retrieved": [
            {"issue_id": r.id, "title": r.title, "body": r.body}
            for r in retrieved
        ]
    })
```

### 2. 生成预测结果

```json
{
  "model_name": "your-model-name",
  "predictions": [
    {
      "query": "TypeError when using async functions",
      "retrieved": [
        {
          "issue_id": "issue_123",
          "title": "Async decorator causes TypeError",
          "body": "Steps to reproduce: 1. Create async function..."
        }
      ]
    }
  ]
}
```

### 3. 运行评估

```bash
python eval.py --submission predictions.json --api-key YOUR_OPENAI_KEY
```

## 评估指标

使用 LLM-as-Judge 评估：

| 指标 | 说明 |
|------|------|
| **Relevance** | 相关性评分 (1-5) |
| **Usefulness** | 对解决问题的有用程度 (1-5) |

## 输出示例

```json
{
  "task": "github_issues",
  "model_name": "your-model",
  "avg_relevance": 3.9,
  "avg_usefulness": 3.5,
  "high_relevance_ratio": 0.62,
  "num_queries": 500
}
```

## 参考资料

- [数据集页面](https://huggingface.co/datasets/lewtun/github-issues)
