# 搜索与检索任务数据集

本仓库包含用于评估搜索和检索系统的数据集及任务定义。每个任务都有明确的目标和评估方法，可用于测试和改进 AI 系统的搜索、检索和理解能力。

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
│   ├── README.md            # 任务总览
│   ├── squad2_reading_comprehension.md
│   ├── chartqa_visual_reasoning.md
│   └── ...
├── datasets/                 # 数据集存储
│   ├── text/                # 文本数据集
│   ├── multimodal/          # 多模态数据集
│   ├── audio/               # 音频数据集
│   ├── video/               # 视频数据集
│   └── ...
├── scripts/                  # 数据下载脚本
│   └── download_datasets.py
├── pyproject.toml           # 项目配置
└── uv.lock                  # 依赖锁定
```

## 快速开始

### 安装依赖

```bash
# 使用 uv 管理项目
uv sync

# 或使用 pip
pip install -e .
```

### 下载数据集

```bash
# 下载所有数据集
python scripts/download_datasets.py

# 数据集会自动保存到 datasets/ 目录
```

### 选择任务

1. 浏览 `tasks/` 目录，选择感兴趣的任务
2. 阅读任务描述，了解目标和评估方法
3. 加载对应的数据集
4. 实现你的解决方案
5. 使用提供的评估脚本测试性能

## 任务格式

每个任务文件包含：

- **任务描述**：需要完成的目标
- **数据集信息**：数据来源、规模、字段说明
- **评估目标**：具体的性能指标要求
- **评估方法**：包含可运行的评估代码
- **建议方案**：推荐的解决思路
- **核心挑战**：主要难点说明

## 评估类型

| 评估方式 | 说明 | 适用场景 |
|----------|------|----------|
| **精确匹配 (EM)** | 预测与标准答案完全一致 | 问答任务 |
| **F1 分数** | 词/字符级别的重叠程度 | 问答、抽取任务 |
| **Recall@K** | 正确答案在前 K 个结果中的比例 | 检索任务 |
| **执行准确率** | 生成的代码执行结果正确 | SQL、程序生成 |
| **LLM-as-Judge** | 使用大模型评估质量 | 无标准答案的场景 |

## 贡献指南

欢迎贡献新的任务定义或改进现有任务：

1. Fork 本仓库
2. 创建新任务文件 `tasks/your_task.md`
3. 遵循现有任务的格式
4. 提交 Pull Request

## 许可证

各数据集遵循其原始许可证，详见各数据集的 README 文件。
