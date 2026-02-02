# 任务列表

本目录包含 15 个独立的搜索与检索评估任务。每个任务目录都是完全自包含的，包含独立的评估脚本和提交模板。

## 目录结构

每个任务目录包含：
```
task_name/
├── README.md           # 任务说明（中文）
├── eval.py             # 独立评估脚本
├── example.json        # 提交示例
└── predictions.json    # 空的预测模板
```

## 任务总览

### 文本问答

| 任务 | 目录 | 数据集 | 评估方式 |
|------|------|--------|----------|
| SQuAD 2.0 阅读理解 | `squad2/` | `rajpurkar/squad_v2` | EM/F1 |
| CMRC 2018 中文阅读理解 | `cmrc2018/` | `hfl/cmrc2018` | EM/F1 |

### 多模态

| 任务 | 目录 | 数据集 | 评估方式 |
|------|------|--------|----------|
| ChartQA 图表推理 | `chartqa/` | `HuggingFaceM4/ChartQA` | 宽松准确率 |
| COCO 图文检索 | `coco/` | `nlphuji/mscoco_2014_5k_test` | R@K |
| AudioCaps 音频检索 | `audiocaps/` | `AudioLLMs/audiocaps_test` | R@K |
| MSVD 视频检索 | `msvd/` | `friedrichor/MSVD` | R@K |

### 领域任务

| 任务 | 目录 | 数据集 | 评估方式 |
|------|------|--------|----------|
| FinQA 金融推理 | `finqa/` | `dreamerdeo/finqa` | 执行准确率 |
| Spider Text-to-SQL | `spider/` | `xlangai/spider` | 执行准确率 |
| CUAD 合同审查 | `cuad/` | `theatticusproject/cuad-qa` | AUPR/F1 |

### RAG 与问答

| 任务 | 目录 | 数据集 | 评估方式 |
|------|------|--------|----------|
| MultiHop-RAG | `multihop_rag/` | `yixuantt/MultiHopRAG` | Recall/F1 |
| ELI5 问答检索 | `eli5/` | `Pavithree/eli5` | MRR/NDCG |

### 对话与社交

| 任务 | 目录 | 数据集 | 评估方式 |
|------|------|--------|----------|
| WildChat 对话检索 | `wildchat/` | `sam-paech/wildchat_*` | LLM-as-Judge |
| Discord 聊天检索 | `discord/` | `breadlicker45/discord-chat` | LLM-as-Judge |
| GitHub Issues 检索 | `github_issues/` | `lewtun/github-issues` | LLM-as-Judge |
| Enron 邮件搜索 | `enron/` | `SetFit/enron_spam` | F1/LLM-as-Judge |

## 快速开始

1. 进入任务目录
```bash
cd tasks/squad2
```

2. 阅读任务说明
```bash
cat README.md
```

3. 查看提交示例
```bash
cat example.json
```

4. 填写预测结果
```bash
# 编辑 predictions.json
```

5. 运行评估
```bash
python eval.py --submission predictions.json
```

## 评估方式说明

| 评估方式 | 说明 |
|----------|------|
| **EM (Exact Match)** | 预测与标准答案完全一致 |
| **F1** | 词/字符级别的重叠程度 |
| **R@K (Recall@K)** | 正确答案在前 K 个结果中的比例 |
| **MRR** | 平均倒数排名 |
| **NDCG** | 归一化折损累积增益 |
| **执行准确率** | 生成代码执行结果正确 |
| **AUPR** | 精确率-召回率曲线下面积 |
| **LLM-as-Judge** | 使用大模型评估质量 |
