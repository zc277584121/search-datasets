# 任务：Spider 自然语言转 SQL

## 任务描述

构建一个自然语言到 SQL 的系统，能够将用户问题翻译成可执行的 SQL 查询，并且能够泛化到**未见过的数据库模式**。系统必须能够处理新领域和复杂的查询模式。

## 数据集信息

| 属性 | 值 |
|------|-----|
| **名称** | Spider |
| **来源** | `xlangai/spider` |
| **语言** | 英语 |
| **许可证** | CC BY-SA 4.0 |
| **规模** | 训练集 7K，验证集 1K，200 个数据库 |

### 数据字段

| 字段 | 类型 | 描述 |
|------|------|------|
| `db_id` | string | 数据库标识符 |
| `question` | string | 自然语言问题 |
| `query` | string | 标准 SQL 查询 |
| `query_toks` | list | SQL 分词结果 |

### SQL 复杂度级别

| 级别 | 占比 | 描述 |
|------|------|------|
| Easy | 25% | 单表、简单条件 |
| Medium | 40% | 多条件、简单 JOIN |
| Hard | 20% | 嵌套查询、多表 JOIN |
| Extra Hard | 15% | 复杂嵌套、集合运算 |

### 零样本设置

测试集的数据库在训练时**从未出现**——系统必须能够泛化到新的模式。

## 评估目标

达到：
- **执行准确率 ≥ 70%**：生成的 SQL 产生正确结果
- **精确匹配 ≥ 60%**：SQL 结构匹配（忽略值）

## 评估方法

### 执行准确率

执行预测的和标准的 SQL，比较结果。

```python
import sqlite3

def execute_sql(db_path, sql):
    """执行 SQL 并返回结果"""
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        cursor.execute(sql)
        results = cursor.fetchall()
        conn.close()
        return results
    except Exception as e:
        return f"ERROR: {e}"

def execution_accuracy(pred_sql, gold_sql, db_path):
    """检查预测的 SQL 是否产生与标准相同的结果"""
    pred_results = execute_sql(db_path, pred_sql)
    gold_results = execute_sql(db_path, gold_sql)

    if isinstance(pred_results, str) and pred_results.startswith("ERROR"):
        return False

    # 作为集合比较（顺序无关）
    return set(map(tuple, pred_results)) == set(map(tuple, gold_results))
```

### 精确匹配（组件匹配）

```python
import re

def normalize_sql(sql):
    """标准化 SQL 以便比较"""
    sql = sql.lower()
    sql = re.sub(r'\s+', ' ', sql)
    sql = re.sub(r'"([^"]*)"', r"'\1'", sql)  # 标准化引号
    return sql.strip()

def sql_exact_match(pred_sql, gold_sql):
    """检查结构匹配（忽略字面值）"""
    # 这是简化版本——完整实现需要 SQL 解析
    pred_normalized = normalize_sql(pred_sql)
    gold_normalized = normalize_sql(gold_sql)
    return pred_normalized == gold_normalized

def evaluate(predictions, dataset, db_dir):
    """
    predictions: 预测的 SQL 字符串列表
    dataset: 包含 'query' 和 'db_id' 的样本列表
    db_dir: 数据库文件目录路径
    """
    exec_correct = 0
    em_correct = 0

    for pred, example in zip(predictions, dataset):
        gold = example['query']
        db_path = f"{db_dir}/{example['db_id']}/{example['db_id']}.sqlite"

        if execution_accuracy(pred, gold, db_path):
            exec_correct += 1
        if sql_exact_match(pred, gold):
            em_correct += 1

    return {
        'execution_accuracy': 100.0 * exec_correct / len(dataset),
        'exact_match': 100.0 * em_correct / len(dataset)
    }
```

## 建议方案

1. **模式感知提示**：在 LLM 提示中包含完整模式

2. **上下文学习**：从相似数据库中选择少量示例

3. **微调模型**：在 SQL 上微调的 CodeLlama、StarCoder

4. **分解策略**：将复杂查询分解为子查询

## 核心挑战

- 理解数据库模式关系
- 生成正确的 JOIN 和表别名
- 处理嵌套查询和集合运算
- 列名消歧
- 值预测（特别是过滤条件）

## 模式格式

每个数据库的模式包含：
- 表名及其列
- 主键和外键
- 列类型（text、number 等）

```json
{
  "db_id": "concert_singer",
  "table_names": ["singer", "concert", "singer_in_concert"],
  "column_names": [
    [-1, "*"],
    [0, "singer_id"],
    [0, "name"],
    [1, "concert_id"],
    [1, "year"],
    [2, "singer_id"],
    [2, "concert_id"]
  ],
  "primary_keys": [1, 3],
  "foreign_keys": [[5, 1], [6, 3]]
}
```

## 参考资料

- 排行榜: https://yale-lily.github.io/spider
- 论文: [Spider: A Large-Scale Human-Labeled Dataset for Complex and Cross-Domain Semantic Parsing and Text-to-SQL Task](https://aclanthology.org/D18-1425/)
