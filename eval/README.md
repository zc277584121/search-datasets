# 评估框架

本目录包含标准化的评估脚本，用于评估各任务的提交结果。

## 目录结构

```
eval/
├── README.md           # 本文件
├── run_eval.py         # 统一评估入口
├── metrics.py          # 通用评估指标
├── llm_judge.py        # LLM-as-Judge 评估
└── configs/            # 各任务的评估配置
    ├── squad2.yaml
    ├── chartqa.yaml
    └── ...
```

## 使用方法

### 运行评估

```bash
# 评估单个任务
python eval/run_eval.py --task squad2 --submission submissions/squad2/predictions.json

# 评估所有任务
python eval/run_eval.py --all --submission-dir submissions/
```

### 提交格式

每个任务的提交格式在 `submissions/<task_name>/` 目录中有模板说明。

## 评估指标

| 指标 | 说明 | 适用任务 |
|------|------|----------|
| `exact_match` | 精确匹配 | 问答任务 |
| `f1` | 词级/字符级 F1 | 问答任务 |
| `recall_at_k` | Top-K 召回率 | 检索任务 |
| `mrr` | 平均倒数排名 | 检索任务 |
| `execution_accuracy` | 执行准确率 | SQL/程序任务 |
| `llm_relevance` | LLM 相关性评分 | 无标准答案任务 |

## 输出格式

评估结果以 JSON 格式输出：

```json
{
  "task": "squad2",
  "metrics": {
    "exact_match": 72.5,
    "f1": 81.3
  },
  "num_samples": 11873,
  "timestamp": "2024-01-30T12:00:00"
}
```
