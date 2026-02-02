# SQuAD 2.0 阅读理解

## 任务描述

SQuAD 2.0 是斯坦福问答数据集的升级版，包含可回答和不可回答的问题，评估模型的阅读理解能力。

## 数据集信息

- **来源**: `rajpurkar/squad_v2`
- **评测集**: 500 条
- **语言**: 英语

## 数据格式

### queries.json 字段说明

| 字段 | 类型 | 说明 |
|------|------|------|
| `id` | string | 问题唯一标识符 |
| `context` | string | 上下文段落 |
| `question` | string | 问题 |

### 预测结果格式

```json
{
  "model_name": "your-model-name",
  "predictions": {
    "question_id": "answer text or empty string"
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
| **Exact Match** | 精确匹配率 |
| **F1** | F1 分数 |

## 参考资料

- [SQuAD 2.0](https://rajpurkar.github.io/SQuAD-explorer/)
