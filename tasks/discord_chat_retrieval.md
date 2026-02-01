# 任务：Discord 聊天消息检索

## 任务描述

构建一个即时通讯消息检索系统，能够从 Discord 聊天记录中检索相关消息。系统需要处理非正式、口语化的对话风格。

## 数据集信息

| 属性 | 值 |
|------|-----|
| **名称** | Discord Chat |
| **来源** | `breadlicker45/discord-chat` |
| **语言** | 英语为主 |
| **许可证** | 见原始数据集 |
| **规模** | 11,136 条消息 |

### 数据字段

| 字段 | 类型 | 描述 |
|------|------|------|
| `data` | string | 聊天消息文本 |

### 数据特点

- 非正式语言风格
- 大量俚语、缩写、表情符号
- 短消息为主
- 可能包含游戏/技术相关讨论

## 评估目标

由于没有标注的检索标签，采用 **LLM-as-Judge** 方式评估：
- **检索相关性 ≥ 3.5/5**：检索结果与查询的语义相关程度
- **风格匹配度**：检索结果是否符合 Discord 社区风格

## 评估方法

### LLM-as-Judge 评估

```python
from openai import OpenAI

def evaluate_chat_retrieval(query, retrieved_messages, client):
    """
    使用 LLM 评估聊天消息检索相关性

    参数:
        query: 用户查询
        retrieved_messages: 检索到的消息列表
        client: OpenAI 客户端
    """
    results = []

    for msg in retrieved_messages[:10]:  # 评估前 10 条
        prompt = f"""请评估以下 Discord 消息与查询的相关性。

查询：{query}

消息：{msg}

评分标准（1-5分）：
1 = 完全不相关
2 = 主题略微相关但内容不匹配
3 = 主题相关但不是最佳匹配
4 = 高度相关
5 = 完全匹配查询意图

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

        results.append({'message': msg, 'score': score})

    return {
        'mean_relevance': sum(r['score'] for r in results) / len(results),
        'details': results
    }
```

### 语义相似度评估

```python
from sentence_transformers import SentenceTransformer
import numpy as np

def evaluate_semantic_similarity(queries, retrieved_messages, model_name='all-MiniLM-L6-v2'):
    """
    使用语义相似度评估检索质量

    参数:
        queries: 查询列表
        retrieved_messages: 每个查询检索到的消息列表
        model_name: 句向量模型名称
    """
    model = SentenceTransformer(model_name)

    similarities = []
    for query, messages in zip(queries, retrieved_messages):
        query_emb = model.encode(query)
        msg_embs = model.encode(messages[:5])  # 前 5 条

        # 计算相似度
        sims = np.dot(msg_embs, query_emb) / (
            np.linalg.norm(msg_embs, axis=1) * np.linalg.norm(query_emb)
        )
        similarities.append(np.mean(sims))

    return {
        'mean_similarity': np.mean(similarities),
        'std_similarity': np.std(similarities)
    }
```

### 多样性评估

```python
def evaluate_diversity(retrieved_messages):
    """
    评估检索结果的多样性

    参数:
        retrieved_messages: 检索到的消息列表
    """
    # 使用词汇多样性作为指标
    all_words = set()
    total_words = 0

    for msg in retrieved_messages:
        words = msg.lower().split()
        all_words.update(words)
        total_words += len(words)

    type_token_ratio = len(all_words) / total_words if total_words > 0 else 0

    return {
        'vocabulary_diversity': type_token_ratio,
        'unique_words': len(all_words),
        'total_words': total_words
    }
```

## 建议方案

1. **口语化文本向量化**：使用在社交媒体上预训练的模型

2. **缩写/俚语处理**：扩展常见的网络用语缩写

3. **关键词 + 语义混合**：结合精确匹配和语义检索

4. **上下文窗口**：考虑消息的前后文

## 核心挑战

- 非标准拼写和语法
- 大量缩写（lol、brb、gg 等）
- 表情符号和特殊字符
- 上下文依赖性强

## 查询示例

```
查询："有人一起打 valorant 吗"
期望检索到：
- "anyone down for some ranked?"
- "looking for teammates for val"
- "who's playing tonight"
```

## 参考资料

- HuggingFace: https://huggingface.co/datasets/breadlicker45/discord-chat
