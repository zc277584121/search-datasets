# 搜索与检索任务数据集

本仓库包含用于评估搜索和检索系统的数据集及任务定义。每个任务都是完全独立的，包含自己的评估脚本、提交模板和说明文档。

## 任务总览

| 任务 | 类型 | 数据集 | 评估方式 |
|------|------|--------|----------|
| [SQuAD 2.0 阅读理解](tasks/squad2/) | 文本问答 | `rajpurkar/squad_v2` | EM/F1 |
| [CMRC 2018 中文阅读理解](tasks/cmrc2018/) | 中文问答 | `hfl/cmrc2018` | EM/F1 |
| [ChartQA 图表推理](tasks/chartqa/) | 多模态 | `HuggingFaceM4/ChartQA` | 宽松准确率 |
| [COCO 图文检索](tasks/coco/) | 多模态 | `nlphuji/mscoco_2014_5k_test` | R@K |
| [AudioCaps 音频检索](tasks/audiocaps/) | 音频 | `AudioLLMs/audiocaps_test` | R@K |
| [MSVD 视频检索](tasks/msvd/) | 视频 | `friedrichor/MSVD` | R@K |
| [FinQA 金融推理](tasks/finqa/) | 领域 | `dreamerdeo/finqa` | 执行准确率 |
| [Spider Text-to-SQL](tasks/spider/) | 领域 | `xlangai/spider` | 执行准确率 |
| [CUAD 合同审查](tasks/cuad/) | 法律 | `theatticusproject/cuad-qa` | AUPR/F1 |
| [MultiHop-RAG](tasks/multihop_rag/) | RAG | `yixuantt/MultiHopRAG` | Recall/F1 |
| [ELI5 问答检索](tasks/eli5/) | 问答 | `Pavithree/eli5` | MRR/NDCG |
| [WildChat 对话检索](tasks/wildchat/) | 对话 | `sam-paech/wildchat_*` | LLM-as-Judge |
| [Discord 聊天检索](tasks/discord/) | 对话 | `breadlicker45/discord-chat` | LLM-as-Judge |
| [GitHub Issues 检索](tasks/github_issues/) | 代码 | `lewtun/github-issues` | LLM-as-Judge |
| [Enron 邮件搜索](tasks/enron/) | 邮件 | `SetFit/enron_spam` | F1/LLM-as-Judge |

## 目录结构

```
search-datasets/
├── README.md                 # 本文件
├── tasks/                    # 任务目录
│   ├── README.md            # 任务总览
│   ├── squad2/              # 各任务独立目录
│   │   ├── README.md        # 任务说明
│   │   ├── eval.py          # 评估脚本
│   │   ├── example.json     # 提交示例
│   │   └── predictions.json # 待填写的预测文件
│   ├── cmrc2018/
│   ├── chartqa/
│   └── ...                  # 其他任务
├── datasets/                 # 数据集存储（可选）
├── scripts/                  # 工具脚本
└── pyproject.toml
```

## 快速开始

### 1. 安装依赖

```bash
uv sync
# 或
pip install -e .
```

### 2. 选择任务

```bash
# 查看所有任务
ls tasks/

# 进入任务目录
cd tasks/squad2

# 阅读任务说明
cat README.md
```

### 3. 实现模型并生成预测

1. 阅读 `tasks/<任务名>/README.md` 了解任务要求
2. 参考 `example.json` 了解提交格式
3. 实现你的模型/方法
4. 将预测结果写入 `predictions.json`

### 4. 运行评估

```bash
# 在任务目录中运行
python eval.py --submission predictions.json

# 保存评估结果
python eval.py --submission predictions.json --output results.json
```

## 评估类型

| 评估方式 | 说明 | 适用任务 |
|----------|------|----------|
| **精确匹配 (EM)** | 预测与标准答案完全一致 | 问答任务 |
| **F1 分数** | 词/字符级别的重叠程度 | 问答、抽取任务 |
| **Recall@K** | 正确答案在前 K 个结果中的比例 | 检索任务 |
| **MRR/NDCG** | 排序质量指标 | 检索任务 |
| **执行准确率** | 生成的代码执行结果正确 | SQL、程序生成 |
| **LLM-as-Judge** | 使用大模型评估质量 | 无标准答案的场景 |

## 提交格式示例

### 问答任务（如 SQuAD）

```json
{
  "model_name": "my-model",
  "predictions": {
    "question_id_1": "答案文本",
    "question_id_2": ""
  }
}
```

### 检索任务（如 COCO）

```json
{
  "model_name": "my-model",
  "image_to_text": {
    "image_id": ["text_id_1", "text_id_2"]
  },
  "text_to_image": {
    "text_id": ["image_id_1", "image_id_2"]
  }
}
```

### LLM-as-Judge 任务（如 Discord）

```json
{
  "model_name": "my-model",
  "predictions": [
    {
      "query": "查询文本",
      "retrieved": [
        {"message_id": "msg_1", "text": "检索结果1"},
        {"message_id": "msg_2", "text": "检索结果2"}
      ]
    }
  ]
}
```

## 评估输出

所有评估结果以 JSON 格式输出：

```json
{
  "task": "squad2",
  "model_name": "my-model",
  "exact_match": 72.5,
  "f1": 81.3,
  "num_samples": 11873,
  "timestamp": "2024-01-30T12:00:00"
}
```

## 许可证

各数据集遵循其原始许可证，详见各数据集的 README 文件。
