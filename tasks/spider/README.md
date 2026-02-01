# Spider Text-to-SQL

## 任务描述

Spider 是一个跨领域的 Text-to-SQL 数据集，要求模型将自然语言问题转换为 SQL 查询语句。数据集包含 200+ 个数据库，涵盖 138 个不同领域。

## 数据集

- **来源**: `xlangai/spider`
- **规模**: ~10,000 问题-SQL 对
- **语言**: 英语
- **数据库数**: 200+

## 任务目标

给定数据库 schema 和自然语言问题，生成正确的 SQL 查询语句。

## 评估指标

| 指标 | 说明 |
|------|------|
| **执行准确率 (EX)** | SQL 执行结果与标准答案一致的比例 |
| **精确匹配 (EM)** | SQL 语句与标准答案完全匹配的比例 |

评估时考虑 SQL 的语义等价性，而非字符串匹配。

## 提交格式

```json
{
  "model_name": "your-model-name",
  "predictions": {
    "question_id_1": "SELECT name FROM students WHERE age > 20",
    "question_id_2": "SELECT COUNT(*) FROM orders WHERE status = 'completed'"
  }
}
```

## 运行评估

```bash
python eval.py --submission predictions.json
```

## 输出示例

```json
{
  "task": "spider",
  "execution_accuracy": 72.5,
  "exact_match": 65.3,
  "easy_accuracy": 85.2,
  "medium_accuracy": 71.4,
  "hard_accuracy": 58.6,
  "extra_hard_accuracy": 42.1,
  "num_samples": 1034,
  "timestamp": "2024-01-30T12:00:00"
}
```

## 难度分级

Spider 数据集按 SQL 复杂度分为四个级别：
- **Easy**: 单表查询，无 JOIN
- **Medium**: 简单 JOIN，基础聚合
- **Hard**: 多表 JOIN，嵌套子查询
- **Extra Hard**: 复杂嵌套，UNION/EXCEPT

## 参考资料

- [Spider 论文](https://arxiv.org/abs/1809.08887)
- [官方排行榜](https://yale-lily.github.io/spider)
- [官方代码](https://github.com/taoyds/spider)
