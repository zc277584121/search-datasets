# 搜索与检索任务数据集

本仓库包含用于评估搜索和检索系统的数据集及任务定义。每个任务都有明确的目标、标准化的评估脚本和提交格式。

## 任务总览

| 任务 | 类型 | 数据集 | 评估方式 |
|------|------|--------|----------|
| [SQuAD 2.0 阅读理解](tasks/squad2_reading_comprehension.md) | 文本问答 | `rajpurkar/squad_v2` | EM/F1 |
| [CMRC 2018 中文阅读理解](tasks/cmrc2018_chinese_mrc.md) | 中文问答 | `hfl/cmrc2018` | EM/F1 |
| [ChartQA 图表推理](tasks/chartqa_visual_reasoning.md) | 多模态 | `HuggingFaceM4/ChartQA` | 宽松准确率 |
| [COCO 图文检索](tasks/coco_image_text_retrieval.md) | 多模态 | `nlphuji/mscoco_2014_5k_test` | R@K |
| [AudioCaps 音频检索](tasks/audiocaps_audio_retrieval.md) | 音频 | `AudioLLMs/audiocaps_test` | R@K |
| [MSVD 视频检索](tasks/msvd_video_retrieval.md) | 视频 | `friedrichor/MSVD` | R@K |
| [FinQA 金融推理](tasks/finqa_numerical_reasoning.md) | 领域 | `dreamerdeo/finqa` | 执行准确率 |
| [Spider Text-to-SQL](tasks/spider_text_to_sql.md) | 领域 | `xlangai/spider` | 执行准确率 |
| [CUAD 合同审查](tasks/cuad_contract_review.md) | 法律 | `theatticusproject/cuad-qa` | AUPR/F1 |
| [MultiHop-RAG](tasks/multihop_rag.md) | RAG | `yixuantt/MultiHopRAG` | Recall/F1 |
| [ELI5 问答检索](tasks/eli5_qa_retrieval.md) | 问答 | `Pavithree/eli5` | MRR/NDCG |
| [WildChat 对话检索](tasks/wildchat_dialogue_retrieval.md) | 对话 | `sam-paech/wildchat_*` | LLM-as-Judge |
| [Discord 聊天检索](tasks/discord_chat_retrieval.md) | 对话 | `breadlicker45/discord-chat` | LLM-as-Judge |
| [GitHub Issues 检索](tasks/github_issues_retrieval.md) | 代码 | `lewtun/github-issues` | LLM-as-Judge |
| [Enron 邮件搜索](tasks/enron_email_search.md) | 邮件 | `SetFit/enron_spam` | F1/LLM-as-Judge |

## 目录结构

```
search-datasets/
├── README.md                 # 本文件
├── tasks/                    # 任务定义
│   ├── README.md
│   └── *.md                 # 各任务详细说明
├── datasets/                 # 数据集存储
│   └── */                   # 各类数据集
├── eval/                     # 评估脚本
│   ├── run_eval.py          # 统一评估入口
│   ├── metrics.py           # 评估指标
│   └── llm_judge.py         # LLM 评估
├── submissions/              # 提交目录
│   └── <task>/              # 各任务提交模板
│       ├── README.md        # 提交格式说明
│       ├── example.json     # 示例
│       └── predictions.json # 待填写的预测文件
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

### 2. 下载数据集

```bash
python scripts/download_datasets.py
```

### 3. 选择任务并实现

1. 阅读 `tasks/<任务名>.md` 了解任务要求
2. 加载 `datasets/` 中的数据
3. 实现你的模型/方法
4. 生成预测结果

### 4. 填写提交文件

按照 `submissions/<任务名>/README.md` 的格式填写 `predictions.json`

### 5. 运行评估

```bash
# 评估单个任务
python eval/run_eval.py --task squad2 --submission submissions/squad2/predictions.json

# 保存评估结果
python eval/run_eval.py --task squad2 --submission submissions/squad2/predictions.json --output results/squad2.json
```

## 评估类型

| 评估方式 | 说明 | 适用任务 |
|----------|------|----------|
| **精确匹配 (EM)** | 预测与标准答案完全一致 | 问答任务 |
| **F1 分数** | 词/字符级别的重叠程度 | 问答、抽取任务 |
| **Recall@K** | 正确答案在前 K 个结果中的比例 | 检索任务 |
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
    "image_id": ["text_id_1", "text_id_2", ...]
  },
  "text_to_image": {
    "text_id": ["image_id_1", "image_id_2", ...]
  }
}
```

### LLM-as-Judge 任务（如 Discord）

```json
{
  "model_name": "my-model",
  "queries": ["查询1", "查询2"],
  "predictions": [
    {"query": "查询1", "retrieved": ["结果1", "结果2"]}
  ]
}
```

## 评估输出

所有评估结果以 JSON 格式输出：

```json
{
  "task": "squad2",
  "exact_match": 72.5,
  "f1": 81.3,
  "num_samples": 11873,
  "timestamp": "2024-01-30T12:00:00"
}
```

## 许可证

各数据集遵循其原始许可证，详见各数据集的 README 文件。
