# CMRC 2018 中文阅读理解

## 任务描述

CMRC 2018 是中文机器阅读理解数据集，评估模型在中文语境下的阅读理解能力。

## 数据集信息

- **来源**: `hfl/cmrc2018`
- **评测集**: 500 条
- **语言**: 中文

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
    "question_id": "答案文本"
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

## 数据来源

`queries.json` 从 HuggingFace `hfl/cmrc2018` 数据集的 validation split 采样 500 条生成。

原始数据集格式：
```python
{
    "id": "问题ID",
    "context": "上下文段落",
    "question": "问题",
    "answers": {
        "text": ["答案"],
        "answer_start": [起始位置]
    }
}
```

## 参考资料

- [CMRC 2018](https://hfl-rc.com/cmrc2018/)
