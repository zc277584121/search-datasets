# SQuAD 2.0 提交格式

## 任务说明

根据维基百科段落回答问题，对于无法回答的问题返回空字符串。

## 输入

评估时会加载 `datasets/text/squad2/validation.parquet`，包含以下字段：
- `id`: 问题唯一标识符
- `context`: 段落文本
- `question`: 问题
- `answers`: 标准答案（评估时使用）

## 输出格式

提交文件 `predictions.json`，格式如下：

```json
{
  "model_name": "你的模型名称",
  "model_description": "模型简要描述（可选）",
  "predictions": {
    "问题ID1": "预测的答案文本",
    "问题ID2": "预测的答案文本",
    "问题ID3": "",
    ...
  }
}
```

### 字段说明

| 字段 | 类型 | 必填 | 说明 |
|------|------|------|------|
| `model_name` | string | 是 | 模型名称 |
| `model_description` | string | 否 | 模型描述 |
| `predictions` | object | 是 | 问题ID到答案的映射 |

### 注意事项

- 对于**可回答**的问题：返回从 context 中抽取的答案文本
- 对于**不可回答**的问题：返回空字符串 `""`
- 问题 ID 必须与数据集中的 `id` 字段完全匹配

## 评估指标

- **Exact Match (EM)**: 完全匹配率
- **F1 Score**: 词级 F1 分数

## 运行评估

```bash
python eval/run_eval.py --task squad2 --submission submissions/squad2/predictions.json
```

## 示例

查看 `example.json` 了解正确的格式。
