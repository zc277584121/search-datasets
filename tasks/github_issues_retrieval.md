# 任务：GitHub Issues 检索

## 任务描述

构建一个代码问题检索系统，能够从 GitHub Issues 中检索与查询相关的问题报告、功能请求或讨论。

## 数据集信息

| 属性 | 值 |
|------|-----|
| **名称** | GitHub Issues |
| **来源** | `lewtun/github-issues` |
| **语言** | 英语（技术文档） |
| **许可证** | 见原始数据集 |
| **规模** | 3,019 条 Issues/PRs |

### 数据字段

| 字段 | 类型 | 描述 |
|------|------|------|
| `id` | int | Issue 唯一 ID |
| `number` | int | Issue 编号 |
| `title` | string | Issue 标题 |
| `body` | string | Issue 正文 |
| `state` | string | 状态（open/closed） |
| `labels` | list | 标签列表 |
| `comments` | list | 评论内容 |
| `is_pull_request` | bool | 是否为 PR |

### 标签结构

每个标签包含：
- `name`: 标签名称（如 bug、feature、documentation）
- `color`: 标签颜色
- `description`: 标签描述

## 评估目标

使用 **LLM-as-Judge** 评估检索相关性：
- **检索相关性 ≥ 4/5**
- **标签预测准确率 ≥ 70%**（将检索问题转化为标签分类）

## 评估方法

### LLM-as-Judge 检索评估

```python
from openai import OpenAI

def evaluate_issue_retrieval(query, retrieved_issues, client):
    """
    使用 LLM 评估 Issue 检索相关性

    参数:
        query: 用户查询（描述一个问题或需求）
        retrieved_issues: 检索到的 Issue 列表
        client: OpenAI 客户端
    """
    scores = []

    for issue in retrieved_issues[:5]:
        prompt = f"""请评估以下 GitHub Issue 与查询的相关性。

查询：{query}

Issue 标题：{issue['title']}
Issue 内容：{issue['body'][:500] if issue['body'] else 'N/A'}
标签：{[l['name'] for l in issue['labels']] if issue['labels'] else []}
状态：{issue['state']}

评分标准（1-5分）：
1 = 完全不相关
2 = 主题相关但问题不同
3 = 类似问题但不是同一个
4 = 高度相关的问题
5 = 完全相同或直接相关的问题

请只输出一个数字。"""

        response = client.chat.completions.create(
            model="gpt-4",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=5
        )

        try:
            score = int(response.choices[0].message.content.strip())
        except:
            score = 3

        scores.append(score)

    return {
        'mean_relevance': sum(scores) / len(scores),
        'scores': scores
    }
```

### 标签预测评估

```python
from sklearn.metrics import precision_recall_fscore_support

def evaluate_label_prediction(predictions, ground_truth):
    """
    评估标签预测性能（多标签分类）

    参数:
        predictions: 预测的标签列表（每个样本一个列表）
        ground_truth: 真实的标签列表
    """
    # 收集所有可能的标签
    all_labels = set()
    for labels in ground_truth + predictions:
        all_labels.update(labels)
    all_labels = sorted(list(all_labels))

    # 转换为多标签二值矩阵
    def to_binary(label_lists, all_labels):
        return [
            [1 if label in labels else 0 for label in all_labels]
            for labels in label_lists
        ]

    y_true = to_binary(ground_truth, all_labels)
    y_pred = to_binary(predictions, all_labels)

    # 计算指标
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average='micro'
    )

    return {
        'precision': precision * 100,
        'recall': recall * 100,
        'f1': f1 * 100
    }
```

### 重复 Issue 检测

```python
def evaluate_duplicate_detection(predictions, duplicates_ground_truth):
    """
    评估重复 Issue 检测

    参数:
        predictions: 预测的重复 Issue 对
        duplicates_ground_truth: 真实的重复 Issue 对
    """
    pred_set = set(map(tuple, predictions))
    true_set = set(map(tuple, duplicates_ground_truth))

    tp = len(pred_set & true_set)
    fp = len(pred_set - true_set)
    fn = len(true_set - pred_set)

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

    return {
        'precision': precision * 100,
        'recall': recall * 100,
        'f1': f1 * 100
    }
```

## 建议方案

1. **代码感知向量化**：使用 CodeBERT 或类似模型

2. **标题 + 正文联合编码**：结合标题和正文信息

3. **标签辅助检索**：利用标签信息增强检索

4. **评论信息利用**：包含评论中的解决方案

## 核心挑战

- 技术术语和代码片段
- Issue 描述质量参差不齐
- 重复 Issue 检测
- 跨项目知识迁移

## 查询示例

```
查询："TypeError when parsing JSON response"

期望检索到：
- 包含 "TypeError" 或 "JSON" 的 Issues
- 标签包含 "bug" 的问题
- 类似错误堆栈的讨论
```

## 参考资料

- HuggingFace: https://huggingface.co/datasets/lewtun/github-issues
