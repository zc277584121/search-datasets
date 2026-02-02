# Quora 重复问题检索

## 任务描述

Quora 重复问题检索任务评估模型识别语义相似问题的能力。给定一个问题，从语料库中检索出与之重复/等价的问题。

## 数据集信息

- **来源**: `glue/qqp` (Quora Question Pairs)
- **评测集**: 500 条 queries
- **语料库**: 1103 个问题
- **语言**: 英语

## 数据格式

### queries.json 字段说明

| 字段 | 类型 | 说明 |
|------|------|------|
| `id` | string | 问题唯一标识符 |
| `question` | string | 问题文本 |

### corpus.json 字段说明

| 字段 | 类型 | 说明 |
|------|------|------|
| `id` | string | 文档唯一标识符 |
| `text` | string | 问题文本 |

### 预测结果格式

```json
{
  "model_name": "your-model-name",
  "predictions": {
    "0": ["corpus_id_1", "corpus_id_2", "corpus_id_3"],
    "1": ["corpus_id_5", "corpus_id_8"]
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
| **MAP** | 平均精度均值 |
| **Recall@K** | Top-K 召回率 |

## 数据来源

`queries.json` 从 HuggingFace `glue/qqp` 数据集的 validation split 采样生成。只选择有重复问题的样本作为 queries。

原始数据集格式：
```python
{
    "question1": "问题1",
    "question2": "问题2",
    "label": 1,  # 1=重复, 0=不重复
    "idx": 样本索引
}
```

## 参考资料

- [Quora Question Pairs](https://www.quora.com/q/quoradata/First-Quora-Dataset-Release-Question-Pairs)
- [GLUE Benchmark](https://gluebenchmark.com/)
