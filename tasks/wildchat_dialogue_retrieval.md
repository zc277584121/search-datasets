# 任务：WildChat 创意写作对话检索

## 任务描述

构建一个对话检索系统，能够从真实的 AI 对话记录中检索相关的创意写作对话，并评估对话质量。

## 数据集信息

| 属性 | 值 |
|------|-----|
| **名称** | WildChat Creative Writing 10K |
| **来源** | `sam-paech/wildchat_creative_writing_annotated_10k` |
| **语言** | 多语言（以英语为主） |
| **许可证** | 见原始数据集 |
| **规模** | 10,000 条对话 |

### 数据字段

| 字段 | 类型 | 描述 |
|------|------|------|
| `conversation_id` | string | 对话唯一标识符 |
| `model` | string | 使用的 AI 模型 |
| `conversation` | list | 对话内容列表 |
| `complexity` | int | 复杂度评分 |
| `creativity` | int | 创意性评分 |
| `quality` | int | 质量评分 |
| `category_free` | string | 自由分类标签 |
| `genre_free` | string | 体裁标签 |
| `is_nsfw` | bool | 是否成人内容 |

### 对话结构

每条消息包含：
- `content`: 消息文本
- `role`: 角色（user/assistant）
- `language`: 消息语言
- `toxic`: 是否含毒性内容

## 评估目标

由于没有标准的检索标签，采用 **LLM-as-Judge** 评估：
- **检索相关性 ≥ 4/5**：检索结果与查询的相关程度
- **对话质量评估准确率 ≥ 80%**：预测质量分与标注分的一致性

## 评估方法

### LLM-as-Judge 检索评估

```python
from openai import OpenAI

def evaluate_retrieval_relevance(query, retrieved_conversations, client):
    """
    使用 LLM 评估检索相关性

    参数:
        query: 用户查询
        retrieved_conversations: 检索到的对话列表
        client: OpenAI 客户端
    """
    scores = []

    for conv in retrieved_conversations:
        # 构建对话摘要
        conv_summary = "\n".join([
            f"{msg['role']}: {msg['content'][:200]}..."
            for msg in conv['conversation'][:3]
        ])

        prompt = f"""请评估以下检索结果与查询的相关性。

查询：{query}

检索到的对话：
{conv_summary}

请给出 1-5 分的相关性评分：
1 = 完全不相关
2 = 略微相关
3 = 部分相关
4 = 较为相关
5 = 高度相关

只输出数字评分。"""

        response = client.chat.completions.create(
            model="gpt-4",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=10
        )

        try:
            score = int(response.choices[0].message.content.strip())
            scores.append(score)
        except:
            scores.append(3)  # 默认中等分数

    return {
        'mean_relevance': sum(scores) / len(scores),
        'scores': scores
    }
```

### 对话质量预测评估

```python
from sklearn.metrics import mean_absolute_error, accuracy_score

def evaluate_quality_prediction(predictions, ground_truth):
    """
    评估对话质量预测

    参数:
        predictions: 预测的质量分数列表
        ground_truth: 标注的质量分数列表
    """
    # 计算 MAE
    mae = mean_absolute_error(ground_truth, predictions)

    # 计算分类准确率（将分数离散化）
    pred_classes = [round(p) for p in predictions]
    gt_classes = [round(g) for g in ground_truth]
    accuracy = accuracy_score(gt_classes, pred_classes)

    # 计算相关系数
    correlation = np.corrcoef(predictions, ground_truth)[0, 1]

    return {
        'mae': mae,
        'accuracy': accuracy * 100,
        'correlation': correlation
    }
```

## 建议方案

1. **对话向量化**：将对话内容转换为向量进行检索

2. **多轮对话理解**：考虑对话的上下文和主题演变

3. **质量评估模型**：训练模型预测对话的质量/创意性

4. **混合检索**：结合关键词和语义检索

## 核心挑战

- 对话长度可变
- 多语言混合
- 主观性的质量评估
- 处理敏感/不当内容

## 检索示例

查询："找一个关于科幻小说创作的高质量对话"

期望检索到：
- 类别包含 "science fiction" 或 "sci-fi"
- quality 评分 ≥ 4
- creativity 评分 ≥ 4

## 参考资料

- HuggingFace: https://huggingface.co/datasets/sam-paech/wildchat_creative_writing_annotated_10k
