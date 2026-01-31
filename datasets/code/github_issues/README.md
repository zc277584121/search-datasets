# GitHub Issues

## 数据集概述

GitHub Issues 数据集包含从 GitHub 仓库收集的 Issue 和 Pull Request 数据。该数据集提供了软件开发过程中的问题报告、功能请求、代码审查等真实场景数据，非常适合代码相关的检索和分析任务。

| 属性 | 值 |
|------|-----|
| **来源** | HuggingFace: `lewtun/github-issues` |
| **样本数** | 3,019 |
| **格式** | Parquet |
| **语言** | 英语（技术文档） |
| **任务类型** | Issue 检索、代码问题分类、软件工程分析 |

## 数据集特点

- **真实开发数据**: 来自实际 GitHub 项目的 Issue 和 PR
- **丰富元数据**: 包含标签、状态、评论、用户信息等
- **结构化数据**: 完整的 GitHub API 数据结构
- **包含评论**: 提供 Issue 的讨论内容

## 数据分割

| 分割 | 样本数 | 文件 |
|------|--------|------|
| train | 3,019 | `train.parquet` |

## 字段说明

### 核心字段

| 字段名 | 类型 | 描述 |
|--------|------|------|
| `id` | int | Issue 唯一 ID |
| `number` | int | Issue 编号（仓库内唯一） |
| `title` | string | Issue 标题 |
| `body` | string | Issue 正文内容 |
| `state` | string | 状态（open/closed） |
| `created_at` | int | 创建时间戳 |
| `updated_at` | int | 更新时间戳 |
| `closed_at` | int | 关闭时间戳（若已关闭） |
| `comments` | list[string] | 评论内容列表 |
| `is_pull_request` | bool | 是否为 Pull Request |

### URL 字段

| 字段名 | 类型 | 描述 |
|--------|------|------|
| `url` | string | API URL |
| `html_url` | string | 网页 URL |
| `repository_url` | string | 仓库 API URL |
| `labels_url` | string | 标签 URL |
| `comments_url` | string | 评论 URL |

### 用户信息字段 (user)

| 子字段 | 类型 | 描述 |
|--------|------|------|
| `login` | string | 用户名 |
| `id` | int | 用户 ID |
| `avatar_url` | string | 头像 URL |
| `type` | string | 用户类型 |

### 标签字段 (labels)

每个标签包含：

| 子字段 | 类型 | 描述 |
|--------|------|------|
| `name` | string | 标签名称 |
| `color` | string | 标签颜色 |
| `description` | string | 标签描述 |

### 其他字段

| 字段名 | 类型 | 描述 |
|--------|------|------|
| `locked` | bool | 是否锁定 |
| `author_association` | string | 作者与仓库的关系 |
| `assignee` | dict | 指派人信息 |
| `assignees` | list | 指派人列表 |
| `milestone` | dict | 里程碑信息 |
| `pull_request` | dict | PR 相关信息（若为 PR） |

## 使用方法

```python
import pandas as pd

# 加载数据集
df = pd.read_parquet("train.parquet")

print(f"数据集大小: {len(df)}")
print(f"Issue 数: {len(df[~df['is_pull_request']])}")
print(f"PR 数: {len(df[df['is_pull_request']])}")

# 查看一条 Issue
sample = df[~df['is_pull_request']].iloc[0]
print(f"标题: {sample['title']}")
print(f"状态: {sample['state']}")
print(f"正文: {sample['body'][:300] if sample['body'] else 'N/A'}...")
print(f"评论数: {len(sample['comments']) if sample['comments'] else 0}")

# 按标签筛选
def has_label(row, label_name):
    labels = row['labels']
    return any(l['name'] == label_name for l in labels) if labels else False

# 筛选 bug 相关 Issue
bugs = df[df.apply(lambda x: has_label(x, 'bug'), axis=1)]
print(f"Bug 相关 Issue: {len(bugs)}")

# 分析 Issue 状态分布
print(df['state'].value_counts())
```

## 适用场景

1. **Issue 检索**: 基于问题描述检索相似 Issue
2. **Bug 分类**: 自动分类 Issue 类型（bug/feature/question）
3. **代码搜索辅助**: 结合代码检索找到相关问题
4. **软件工程研究**: 分析开源项目的开发模式
5. **自动标签**: 训练自动标签推荐系统

## 数据示例

```json
{
  "number": 123,
  "title": "Bug: TypeError when parsing JSON response",
  "body": "## Description\nWhen calling the API endpoint...\n\n## Steps to reproduce\n1. ...",
  "state": "open",
  "labels": [{"name": "bug", "color": "d73a4a"}],
  "comments": [
    "I can reproduce this issue on v2.0.1",
    "Fixed in PR #125"
  ],
  "is_pull_request": false
}
```

## 评测指标

- **检索任务**: R@K, MRR, MAP
- **分类任务**: Precision, Recall, F1
- **标签推荐**: Hit@K, NDCG

## 注意事项

- 部分字段可能为空（如未指派的 assignee）
- 评论内容可能包含代码片段和技术术语
- Issue 状态可能已过时（数据集为快照）

## 许可证

请参考原始数据集的许可证说明：[HuggingFace 页面](https://huggingface.co/datasets/lewtun/github-issues)

## 更新日期

2026-01-30
