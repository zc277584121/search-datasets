# Enron 邮件垃圾分类

## 任务描述

Enron 邮件数据集包含安然公司员工的真实电子邮件，任务是判断邮件是否为垃圾邮件。

## 数据集信息

- **来源**: `SetFit/enron_spam`
- **评测集**: 500 条
- **语言**: 英语
- **标签**: spam（垃圾邮件）/ ham（正常邮件）

## 数据格式

### queries.json 字段说明

| 字段 | 类型 | 说明 |
|------|------|------|
| `id` | string | 邮件唯一标识符 |
| `text` | string | 邮件完整文本 |

### 预测结果格式

```json
{
  "model_name": "your-model-name",
  "predictions": {
    "0": "ham",
    "1": "spam"
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
| **F1** | 垃圾邮件分类的 F1 分数 |
| **Precision** | 精确率 |
| **Recall** | 召回率 |
| **Accuracy** | 准确率 |

## 参考资料

- [Enron Email Dataset](https://www.cs.cmu.edu/~enron/)
