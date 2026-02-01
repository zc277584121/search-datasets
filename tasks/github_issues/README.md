# GitHub Issues 检索

## 任务描述

GitHub Issues 检索任务评估模型在技术问题追踪场景下的检索能力。数据包含真实的 GitHub issue，涵盖 bug 报告、功能请求、讨论等。

## 数据集

- **来源**: `lewtun/github-issues`
- **规模**: ~500,000 issues
- **语言**: 英语
- **特点**: 技术性内容、代码片段、结构化信息

## 任务目标

给定用户查询（问题描述），从 issue 库中检索最相关的 issues。需要处理：
1. 技术术语和代码
2. 错误信息和堆栈跟踪
3. Issue 的结构化元数据（标签、状态等）

## 评估指标

| 指标 | 说明 |
|------|------|
| **LLM-as-Judge** | 使用大语言模型评估检索结果的相关性 |
| **Relevance Score** | 相关性评分 (1-5) |
| **Usefulness Score** | 对解决问题的有用程度 (1-5) |

## 提交格式

```json
{
  "model_name": "your-model-name",
  "predictions": [
    {
      "query": "TypeError when using async functions",
      "retrieved": [
        {"issue_id": "issue_123", "title": "...", "body": "..."},
        {"issue_id": "issue_456", "title": "...", "body": "..."}
      ]
    }
  ]
}
```

## 运行评估

```bash
python eval.py --submission predictions.json --api-key YOUR_API_KEY
```

## 输出示例

```json
{
  "task": "github_issues",
  "avg_relevance": 3.9,
  "avg_usefulness": 3.5,
  "high_relevance_ratio": 0.62,
  "num_queries": 100,
  "timestamp": "2024-01-30T12:00:00"
}
```

## Issue 类型

常见的 issue 类型：
- **Bug Report**: 错误报告，包含复现步骤
- **Feature Request**: 功能请求
- **Question**: 使用问题
- **Discussion**: 技术讨论

## 参考资料

- [数据集页面](https://huggingface.co/datasets/lewtun/github-issues)
