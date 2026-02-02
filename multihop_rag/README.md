# MultiHop-RAG 多跳问答

## 任务描述

MultiHop-RAG 评估模型进行多跳推理和检索的能力，需要整合多个文档的信息。

## 数据集信息

- **来源**: `yixuantt/MultiHopRAG`
- **评测集**: 500 条
- **语言**: 英语

## 数据格式

### queries.json 字段说明

| 字段 | 类型 | 说明 |
|------|------|------|
| `id` | string | 问题唯一标识符 |
| `question` | string | 多跳问题 |
| `question_type` | string | 问题类型 |

### 预测结果格式

```json
{
  "model_name": "your-model-name",
  "predictions": {
    "0": {"answer": "Yes", "retrieved_docs": ["doc_1", "doc_2"]}
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
| **Exact Match** | 答案精确匹配率 |
| **F1** | F1 分数 |
| **Retrieval Recall** | 检索召回率 |

## 数据来源

`queries.json` 从 HuggingFace `yixuantt/MultiHopRAG` 数据集采样 500 条生成。

原始数据集格式：
```python
{
    "id": "问题ID",
    "question": "多跳问题",
    "answer": "答案",
    "question_type": "comparison_query/temporal_query/...",
    "evidence_list": [  # 支撑证据
        {"source": "来源", "fact": "事实", "title": "标题"}
    ]
}
```

## 参考资料

- [MultiHop-RAG](https://github.com/yixuantt/MultiHop-RAG)
