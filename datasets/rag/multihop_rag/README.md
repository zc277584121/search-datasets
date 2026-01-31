# MultiHop-RAG 数据集

## 概述

**MultiHop-RAG** 是专门为评估**多跳检索增强生成**（Multi-hop Retrieval-Augmented Generation）设计的数据集，每个问题需要从多个文档中检索和整合信息才能回答。

- **发布机构**: HKUST & Microsoft
- **论文**: "MultiHop-RAG: Benchmarking Retrieval-Augmented Generation for Multi-Hop Queries" (COLM 2024)
- **HuggingFace**: `yixuantt/MultiHopRAG`
- **许可证**: MIT
- **语言**: 英语

## 数据集特点

- **多跳查询**: 每个问题需要 2-4 个文档的信息
- **真实新闻**: 基于英语新闻文章构建
- **元数据丰富**: 包含发布时间、作者等元数据
- **证据标注**: 标注了每个问题的支撑文档

## 数据集规模

| 内容 | 数量 |
|------|------|
| 查询数 | 2,556 |
| 知识库文档数 | 609 |
| 平均证据文档数/查询 | 2-4 |

## 多跳类型

| 类型 | 说明 | 示例 |
|------|------|------|
| **推理型** | 需要逻辑推理连接多个事实 | A 是 B，B 是 C，所以 A 是 C |
| **比较型** | 需要比较多个实体的属性 | X 和 Y 哪个更大？ |
| **桥接型** | 通过中间实体连接 | A 的 CEO 的母校是？ |
| **组合型** | 需要整合多个独立事实 | 列出所有满足条件 X 和 Y 的项 |

## 数据字段说明

### 查询数据 (queries)

| 字段名 | 类型 | 说明 |
|--------|------|------|
| `query_id` | string | 查询唯一标识符 |
| `query` | string | 自然语言问题 |
| `answer` | string | 标准答案 |
| `question_type` | string | 问题类型 |
| `evidence_list` | list | 支撑文档 ID 列表 |

### 知识库数据 (corpus)

| 字段名 | 类型 | 说明 |
|--------|------|------|
| `doc_id` | string | 文档唯一标识符 |
| `title` | string | 新闻标题 |
| `content` | string | 新闻正文 |
| `published_at` | string | 发布时间 |
| `source` | string | 新闻来源 |
| `author` | string | 作者（如有） |
| `url` | string | 原始链接 |
| `category` | string | 新闻类别 |

## 数据示例

### 查询示例

```json
{
  "query_id": "q_001",
  "query": "What company did the CEO of Twitter acquire in 2022, and what was its previous valuation?",
  "answer": "Elon Musk acquired Twitter for $44 billion",
  "question_type": "bridge",
  "evidence_list": ["doc_123", "doc_456"]
}
```

### 文档示例

```json
{
  "doc_id": "doc_123",
  "title": "Elon Musk completes Twitter acquisition",
  "content": "Elon Musk has completed his $44 billion acquisition of Twitter...",
  "published_at": "2022-10-28",
  "source": "Reuters",
  "category": "technology"
}
```

## 评测指标

### 检索评测
| 指标 | 说明 |
|------|------|
| Recall@K | Top-K 结果中包含所有证据文档的比例 |
| MRR | 第一个相关文档排名的倒数平均 |
| MAP | 平均精度均值 |

### 生成评测
| 指标 | 说明 |
|------|------|
| EM (Exact Match) | 完全匹配率 |
| F1 | Token 级 F1 分数 |
| Answer Recall | 答案覆盖率 |

## 使用方法

```python
from datasets import load_dataset

# 加载数据集
dataset = load_dataset("yixuantt/MultiHopRAG")

# 查看数据结构
print(dataset)

# 获取查询
queries = dataset["queries"]
print(queries[0])

# 获取知识库
corpus = dataset["corpus"]
print(corpus[0])

# 构建检索系统
from sentence_transformers import SentenceTransformer

model = SentenceTransformer('all-MiniLM-L6-v2')

# 编码文档
doc_embeddings = model.encode([doc["content"] for doc in corpus])

# 编码查询并检索
query_embedding = model.encode(queries[0]["query"])
# ... 进行相似度检索
```

## RAG Pipeline 评测流程

```
1. 接收查询
   ↓
2. 检索相关文档（需要多个）
   ↓
3. 将查询 + 检索文档送入 LLM
   ↓
4. 生成答案
   ↓
5. 与标准答案比较
```

## 挑战与难点

1. **证据分散**: 答案线索分布在多个文档中
2. **推理链**: 需要正确的推理顺序
3. **噪声干扰**: 知识库中有大量无关文档
4. **元数据利用**: 需要利用时间、来源等元数据

## 应用场景

1. **复杂问答系统**: 处理需要多步推理的问题
2. **研究助手**: 从多个文献中整合信息
3. **事实核查**: 验证需要多个来源的声明
4. **智能搜索**: 回答需要综合多个搜索结果的问题

## 与其他 RAG 数据集比较

| 数据集 | 查询数 | 多跳 | 领域 |
|--------|--------|------|------|
| **MultiHop-RAG** | 2.5K | ✅ 2-4跳 | 新闻 |
| Natural Questions | 300K | ❌ | Wikipedia |
| HotpotQA | 113K | ✅ 2跳 | Wikipedia |
| MuSiQue | 25K | ✅ 2-4跳 | Wikipedia |

## 参考链接

- 论文: https://arxiv.org/abs/2401.15391
- GitHub: https://github.com/yixuantt/MultiHop-RAG
- HuggingFace: https://huggingface.co/datasets/yixuantt/MultiHopRAG
- COLM 2024: https://colmweb.org/
