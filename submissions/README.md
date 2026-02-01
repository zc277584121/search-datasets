# 提交指南

本目录用于存放各任务的解决方案提交。

## 目录结构

```
submissions/
├── README.md                    # 本文件
├── squad2/                      # SQuAD 2.0 任务提交
│   ├── README.md               # 提交格式说明
│   ├── predictions.json        # 你的预测结果（需要填写）
│   └── example.json            # 示例格式
├── chartqa/
├── coco/
└── ...
```

## 提交流程

1. **选择任务**：进入对应任务目录
2. **阅读说明**：查看该任务的 `README.md` 了解提交格式
3. **填写预测**：按照格式填写 `predictions.json`
4. **运行评估**：使用评估脚本验证结果

```bash
# 评估你的提交
python eval/run_eval.py --task squad2 --submission submissions/squad2/predictions.json

# 查看评估结果
python eval/run_eval.py --task squad2 --submission submissions/squad2/predictions.json --output results/squad2_eval.json
```

## 通用提交格式

所有任务的提交都使用 JSON 格式，基本结构如下：

```json
{
  "model_name": "你的模型名称",
  "model_description": "模型简要描述",
  "predictions": {
    // 具体预测内容，格式因任务而异
  }
}
```

## 注意事项

- 确保 JSON 格式正确（可使用 `python -m json.tool` 验证）
- 预测 ID 必须与数据集中的 ID 完全匹配
- 不要修改示例文件，复制后再修改
