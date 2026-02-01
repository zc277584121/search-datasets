# ChartQA 提交格式

## 任务说明

回答关于图表图片的问题，需要视觉理解和数值推理能力。

## 输入

评估时会加载 `datasets/multimodal/chartqa/test.parquet`，包含以下字段：
- `image`: 图表图片
- `query`: 问题
- `label`: 标准答案
- `source`: 问题来源（human/augmented）

## 输出格式

提交文件 `predictions.json`，格式如下：

```json
{
  "model_name": "你的模型名称",
  "model_description": "模型简要描述（可选）",
  "predictions": {
    "0": "45.2",
    "1": "increasing",
    "2": "2020",
    ...
  }
}
```

### 字段说明

| 字段 | 类型 | 必填 | 说明 |
|------|------|------|------|
| `predictions` | object | 是 | 样本索引到答案的映射 |

### 注意事项

- 索引从 0 开始，对应数据集中的行号
- 数值答案：保持适当精度（如 "45.2" 而非 "45.19999"）
- 文本答案：与标准答案大小写无关

## 评估指标

- **Relaxed Accuracy**: 数值答案允许 5% 误差，文本答案完全匹配
- **Human Accuracy**: human 类问题的准确率
- **Augmented Accuracy**: augmented 类问题的准确率

## 运行评估

```bash
python eval/run_eval.py --task chartqa --submission submissions/chartqa/predictions.json
```
