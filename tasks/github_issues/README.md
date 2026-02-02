# GitHub Issues 检索

## 任务描述

GitHub Issues 检索任务评估模型在技术问题追踪场景下的检索能力。数据包含真实的 GitHub issue，涵盖 bug 报告、功能请求、讨论等。

## 数据集信息

- **来源**: `lewtun/github-issues`
- **评测集**: 500 条
- **语言**: 英语

## 数据格式

### queries.json 字段说明

| 字段 | 类型 | 说明 |
|------|------|------|
| `id` | string | Issue 唯一标识符 |
| `title` | string | Issue 标题 |
| `body` | string | Issue 正文描述 |

### 预测结果格式

```json
{
  "model_name": "your-model-name",
  "predictions": [
    {
      "query": "Issue title",
      "retrieved": [
        {"title": "Related issue title", "body": "Issue body..."}
      ]
    }
  ]
}
```

## 快速开始

1. 打开 `run_demo.py`，找到 `# TODO` 注释，替换为你的模型代码
2. 运行：
   ```bash
   python run_demo.py
   ```

## 评估指标

使用 LLM-as-Judge (gpt-4o-mini) 评估：

| 指标 | 说明 |
|------|------|
| **Relevance** | 相关性评分 (1-5) |
| **Usefulness** | 对解决问题的有用程度 (1-5) |

## 参考资料

- [数据集页面](https://huggingface.co/datasets/lewtun/github-issues)
