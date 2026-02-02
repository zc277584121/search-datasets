# ELI5 长文本问答检索

## 任务描述

ELI5 (Explain Like I'm 5) 评估模型为复杂问题检索相关支撑文档的能力。

## 数据集信息

- **来源**: `Pavithree/eli5`
- **评测集**: 500 条
- **语言**: 英语

## 数据格式

### queries.json 字段说明

| 字段 | 类型 | 说明 |
|------|------|------|
| `id` | string | 问题唯一标识符 |
| `question` | string | 问题标题 |
| `selftext` | string | 问题补充说明 |

### 预测结果格式

```json
{
  "model_name": "your-model-name",
  "predictions": {
    "question_id": ["doc_id_1", "doc_id_2", "doc_id_3"]
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
| **MRR** | 平均倒数排名 |
| **NDCG@K** | 归一化折损累积增益 |
| **Recall@K** | 召回率 |

## 参考资料

- [ELI5 Dataset](https://facebookresearch.github.io/ELI5/)
