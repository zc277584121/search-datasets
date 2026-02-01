# Spider Text-to-SQL 提交格式

## 任务说明

将自然语言问题翻译成 SQL 查询语句。

## 输入

评估数据 `datasets/table/spider/`：
- `db_id`: 数据库标识符
- `question`: 自然语言问题
- `query`: 标准 SQL（评估时使用）

数据库文件位于：`datasets/table/spider/database/{db_id}/{db_id}.sqlite`

## 输出格式

提交文件 `predictions.json`，格式如下：

```json
{
  "model_name": "你的模型名称",
  "model_description": "模型简要描述",
  "predictions": [
    {
      "db_id": "concert_singer",
      "question": "How many singers do we have?",
      "predicted_sql": "SELECT count(*) FROM singer"
    },
    {
      "db_id": "concert_singer",
      "question": "What are the names of singers?",
      "predicted_sql": "SELECT name FROM singer"
    }
  ]
}
```

### 字段说明

| 字段 | 类型 | 说明 |
|------|------|------|
| `db_id` | string | 数据库标识符（与输入一致） |
| `question` | string | 原始问题（与输入一致） |
| `predicted_sql` | string | 预测的 SQL 语句 |

### 注意事项

- SQL 语句需要是可执行的
- 保持预测顺序与验证集一致
- 不要包含分号结尾

## 评估指标

- **Execution Accuracy**: SQL 执行结果与标准一致
- **Exact Match**: SQL 结构完全匹配

## 运行评估

```bash
python eval/run_eval.py --task spider --submission submissions/spider/predictions.json
```
