# CMRC 2018 中文阅读理解

## 任务描述

CMRC 2018 是中文机器阅读理解数据集，由哈工大讯飞联合实验室发布。任务要求从给定的中文段落中抽取答案来回答问题。

## 数据集

- **来源**: `hfl/cmrc2018`
- **规模**: ~20,000 问答对
- **语言**: 中文
- **特点**: 所有问题都有答案（与 SQuAD 1.0 类似）

## 任务目标

给定一个中文段落和问题，从段落中抽取能够回答问题的文本片段。

## 评估指标

| 指标 | 说明 |
|------|------|
| **Exact Match (EM)** | 预测答案与标准答案完全匹配的比例 |
| **F1** | 预测答案与标准答案在字符级别的 F1 分数 |

注意：中文评估在字符级别进行，而非词级别。

## 提交格式

```json
{
  "model_name": "your-model-name",
  "predictions": {
    "question_id_1": "答案文本",
    "question_id_2": "另一个答案"
  }
}
```

## 运行评估

```bash
python eval.py --submission predictions.json
```

## 输出示例

```json
{
  "task": "cmrc2018",
  "exact_match": 65.3,
  "f1": 84.7,
  "num_samples": 3219,
  "timestamp": "2024-01-30T12:00:00"
}
```

## 参考资料

- [CMRC 2018 论文](https://arxiv.org/abs/1810.07366)
- [哈工大讯飞联合实验室](https://github.com/ymcui/cmrc2018)
