# Spider Text-to-SQL

## 任务描述

Spider 是跨域 Text-to-SQL 数据集，评估模型将自然语言问题转换为 SQL 查询的能力。

## 数据集信息

- **来源**: `xlangai/spider`
- **评测集**: 1034 条
- **语言**: 英语

## 数据格式

### queries.json 字段说明

| 字段 | 类型 | 说明 |
|------|------|------|
| `id` | string | 问题唯一标识符 |
| `question` | string | 自然语言问题 |
| `db_id` | string | 数据库标识符 |

### 预测结果格式

```json
{
  "model_name": "your-model-name",
  "predictions": {
    "0": "SELECT COUNT(*) FROM singer"
  }
}
```

## 快速开始

1. 打开 `run_demo.py`，找到 `# TODO` 注释，替换为你的模型代码
2. 运行：
   ```bash
   python run_demo.py
   ```

## 评估指标

| 指标 | 说明 |
|------|------|
| **Exact Match** | SQL 精确匹配率 |
| **Execution Accuracy** | 执行结果匹配率（需要数据库） |

## 数据来源

`queries.json` 从 HuggingFace `xlangai/spider` 数据集的 validation split 全量 1034 条生成。

原始数据集格式：
```python
{
    "db_id": "数据库名",
    "question": "自然语言问题",
    "query": "SELECT ... FROM ...",  # 标准SQL答案
    "difficulty": "easy/medium/hard/extra"
}
```

## 参考资料

- [Spider Dataset](https://yale-lily.github.io/spider)
