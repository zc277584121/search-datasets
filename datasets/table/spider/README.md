# Spider 数据集

## 概述

**Spider** 是耶鲁大学发布的大规模跨域 Text-to-SQL 数据集，是 NL2SQL 领域最重要的基准之一。

- **发布机构**: 耶鲁大学
- **论文**: "Spider: A Large-Scale Human-Labeled Dataset for Complex and Cross-Domain Semantic Parsing and Text-to-SQL Task" (EMNLP 2018)
- **HuggingFace**: `xlangai/spider`
- **许可证**: CC BY-SA 4.0
- **语言**: 英语

## 数据集特点

- **跨域**: 涵盖 138 个不同领域（餐厅、体育、音乐等）
- **复杂 SQL**: 包含嵌套查询、JOIN、GROUP BY、HAVING 等
- **零样本评测**: 测试集的数据库在训练集中从未出现
- **多表查询**: 平均每个数据库有 5.1 个表

## 数据集规模

| 子集 | 问题数 | 数据库数 | 说明 |
|------|--------|----------|------|
| train | 7,000 | 140 | 训练集 |
| validation | 1,034 | 20 | 验证集（常用于评测） |
| test | 2,147 | 40 | 测试集（答案不公开） |

**总计**: 10,181 个问题，200 个数据库，5,693 个唯一 SQL

## SQL 复杂度分布

Spider 按 SQL 复杂度分为四级：

| 难度 | 比例 | 特点 |
|------|------|------|
| Easy | 25% | 单表、无 JOIN、简单条件 |
| Medium | 40% | 多条件、简单 JOIN |
| Hard | 20% | 嵌套查询、多表 JOIN |
| Extra Hard | 15% | 复杂嵌套、集合运算 |

## 数据字段说明

| 字段名 | 类型 | 说明 |
|--------|------|------|
| `db_id` | string | 数据库标识符 |
| `question` | string | 自然语言问题 |
| `query` | string | 对应的 SQL 查询 |
| `query_toks` | list | SQL 的 token 序列 |
| `query_toks_no_value` | list | SQL token（值被替换为占位符） |
| `question_toks` | list | 问题的 token 序列 |

## 数据示例

```json
{
  "db_id": "concert_singer",
  "question": "How many singers do we have?",
  "query": "SELECT count(*) FROM singer",
  "query_toks": ["SELECT", "count", "(", "*", ")", "FROM", "singer"],
  "query_toks_no_value": ["SELECT", "count", "(", "*", ")", "FROM", "singer"],
  "question_toks": ["How", "many", "singers", "do", "we", "have", "?"]
}
```

### 复杂查询示例

```json
{
  "db_id": "concert_singer",
  "question": "What are the names of singers who have concerts in 2014 but not 2015?",
  "query": "SELECT T1.name FROM singer AS T1 JOIN concert AS T2 ON T1.singer_id = T2.singer_id WHERE T2.year = 2014 EXCEPT SELECT T1.name FROM singer AS T1 JOIN concert AS T2 ON T1.singer_id = T2.singer_id WHERE T2.year = 2015"
}
```

## 数据库 Schema

每个数据库包含以下元信息：
- 表名列表
- 每个表的列名和类型
- 主键和外键关系

Schema 信息通常在单独的 `tables.json` 文件中。

## 评测指标

- **Exact Match Accuracy**: SQL 与标准答案完全一致（标准化后）
- **Execution Accuracy**: SQL 执行结果与标准答案相同
- **Component Matching**: 分别评估 SELECT/WHERE/GROUP BY 等子句

## 使用方法

```python
from datasets import load_dataset

# 加载数据集
dataset = load_dataset("xlangai/spider")

# 查看训练集
print(dataset["train"][0])

# 按数据库筛选
concert_data = dataset["train"].filter(
    lambda x: x["db_id"] == "concert_singer"
)
```

## 应用场景

1. **自然语言数据库接口**: 让用户用自然语言查询数据库
2. **BI 智能助手**: 商业智能工具的自然语言查询
3. **数据分析 Copilot**: 辅助数据分析师编写 SQL
4. **LLM SQL 能力评测**: 评估大模型的 SQL 生成能力

## 相关数据集

| 数据集 | 特点 |
|--------|------|
| BIRD | 更大规模、更真实的 SQL |
| WikiSQL | 单表简单 SQL |
| CoSQL | 多轮对话式 SQL |
| SParC | 上下文相关的跨域 SQL |

## 参考链接

- 官网: https://yale-lily.github.io/spider
- GitHub: https://github.com/taoyds/spider
- 论文: https://aclanthology.org/D18-1425/
- Leaderboard: https://yale-lily.github.io/spider
