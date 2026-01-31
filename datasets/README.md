# 搜索数据集合集

本目录包含 10 个精选的搜索/检索相关数据集，涵盖文本、多模态、音频、视频等多种场景。

## 数据集概览

| # | 数据集 | 场景 | 样本量 | 体积 | 格式 |
|---|--------|------|--------|------|------|
| 1 | [SQuAD 2.0](text/squad2/) | 通用文本问答 | 142K | ~21MB | Parquet |
| 2 | [CMRC 2018](chinese/cmrc2018/) | 中文阅读理解 | 14K | ~5MB | Parquet |
| 3 | [FinQA](finance/finqa/) | 金融数值推理 | 8.3K | ~83MB | JSON |
| 4 | [CUAD](document/cuad/) | 法律合同理解 | 26K | ~79MB | JSON |
| 5 | [Spider](table/spider/) | NL2SQL | 8K | ~1MB | Parquet |
| 6 | [ChartQA](multimodal/chartqa/) | 图表问答 | 33K | ~1.3GB | Parquet |
| 7 | [COCO Karpathy](multimodal/coco_karpathy/) | 图文检索 | 123K | ~1.3GB | Parquet |
| 8 | [MSVD](video/msvd/) | 视频描述检索 | 2K | ~1MB | Parquet |
| 9 | [AudioCaps](audio/audiocaps/) | 音频描述检索 | 4.4K | ~1.3GB | Parquet |
| 10 | [MultiHop-RAG](rag/multihop_rag/) | 多跳RAG推理 | 3.2K | ~4MB | Parquet |

### 附加数据集（对话/社交/代码）

| # | 数据集 | 场景 | 样本量 | 格式 |
|---|--------|------|--------|------|
| 11 | [WildChat 10K](conversation/wildchat_10k/) | AI创意写作对话 | 10K | Parquet |
| 12 | [Discord Chat](conversation/discord_chat/) | Discord聊天记录 | 11K | Parquet |
| 13 | [ELI5 Reddit](conversation/eli5_reddit/) | Reddit问答 | 229K | Parquet |
| 14 | [GitHub Issues](code/github_issues/) | GitHub Issue/PR | 3K | Parquet |
| 15 | [Enron Email](document/enron_mini/) | 企业邮件 | 34K | Parquet |

**总体积**: 约 3 GB

## 目录结构

```
datasets/
├── text/                    # 文本问答
│   └── squad2/              # SQuAD 2.0
├── chinese/                 # 中文数据集
│   └── cmrc2018/            # CMRC 2018
├── finance/                 # 金融领域
│   └── finqa/               # FinQA
├── document/                # 文档理解
│   └── cuad/                # CUAD 合同数据集
├── table/                   # 表格/SQL
│   └── spider/              # Spider NL2SQL
├── multimodal/              # 多模态
│   ├── chartqa/             # ChartQA 图表问答
│   └── coco_karpathy/       # COCO 图文检索
├── video/                   # 视频
│   └── msvd/                # MSVD 视频描述
├── audio/                   # 音频
│   └── audiocaps/           # AudioCaps
├── rag/                     # RAG 相关
│   └── multihop_rag/        # MultiHop-RAG
├── conversation/            # 对话/社交
│   ├── wildchat_10k/        # WildChat AI 对话
│   ├── discord_chat/        # Discord 聊天
│   └── eli5_reddit/         # Reddit ELI5 问答
└── code/                    # 代码相关
    └── github_issues/       # GitHub Issues
```

## 使用方法

### 加载 Parquet 格式数据集

```python
import pandas as pd

# 加载 SQuAD 2.0 训练集
squad_train = pd.read_parquet("text/squad2/train.parquet")
print(f"SQuAD 训练集: {len(squad_train)} 样本")

# 加载 ChartQA
chartqa = pd.read_parquet("multimodal/chartqa/train.parquet")
```

### 加载 JSON 格式数据集

```python
import json

# 加载 FinQA
with open("finance/finqa/train.json", "r") as f:
    finqa_train = json.load(f)
print(f"FinQA 训练集: {len(finqa_train)} 样本")
```

### 使用 HuggingFace datasets

```python
from datasets import load_dataset

# 这些数据集也可以直接从 HuggingFace 加载
squad = load_dataset("rajpurkar/squad_v2")
cmrc = load_dataset("hfl/cmrc2018")
spider = load_dataset("xlangai/spider")
```

## 各数据集详情

### 1. [SQuAD 2.0](text/squad2/) (文本问答)
- **任务**: 抽取式问答 + 不可回答问题判断
- **来源**: Wikipedia
- **特点**: 包含 5 万+ 不可回答问题
- **评测指标**: EM, F1
- **详细文档**: [README](text/squad2/README.md)

### 2. [CMRC 2018](chinese/cmrc2018/) (中文阅读理解)
- **任务**: 中文片段抽取式问答
- **来源**: 中文 Wikipedia
- **特点**: 包含 challenging 子集
- **评测指标**: EM, F1
- **详细文档**: [README](chinese/cmrc2018/README.md)

### 3. [FinQA](finance/finqa/) (金融数值推理)
- **任务**: 财报表格+文本的数值推理问答
- **来源**: 上市公司 10-K/10-Q 报告
- **特点**: 标注了完整推理程序
- **评测指标**: Execution Accuracy
- **详细文档**: [README](finance/finqa/README.md)

### 4. [CUAD](document/cuad/) (法律合同理解)
- **任务**: 合同条款识别（41类）
- **来源**: SEC EDGAR 商业合同
- **特点**: 法律专家标注
- **评测指标**: AUPR
- **详细文档**: [README](document/cuad/README.md)

### 5. [Spider](table/spider/) (NL2SQL)
- **任务**: 自然语言转 SQL
- **来源**: 200个数据库，138个领域
- **特点**: 跨域零样本评测
- **评测指标**: Exact Match, Execution Accuracy
- **详细文档**: [README](table/spider/README.md)

### 6. [ChartQA](multimodal/chartqa/) (图表问答)
- **任务**: 图表视觉问答
- **来源**: 网络真实图表
- **特点**: 需要视觉+逻辑推理
- **评测指标**: Relaxed Accuracy
- **详细文档**: [README](multimodal/chartqa/README.md)

### 7. [COCO Karpathy](multimodal/coco_karpathy/) (图文检索)
- **任务**: 图像-文本双向检索
- **来源**: MS COCO
- **特点**: 标准图文检索基准
- **评测指标**: R@1, R@5, R@10
- **详细文档**: [README](multimodal/coco_karpathy/README.md)

### 8. [MSVD](video/msvd/) (视频描述检索)
- **任务**: 视频描述生成/检索
- **来源**: YouTube 短视频
- **特点**: 每视频约 40 条描述
- **评测指标**: BLEU, CIDEr, R@K
- **详细文档**: [README](video/msvd/README.md)

### 9. [AudioCaps](audio/audiocaps/) (音频描述检索)
- **任务**: 音频描述生成/检索
- **来源**: AudioSet (YouTube)
- **特点**: 10秒音频片段
- **评测指标**: BLEU, CIDEr, SPIDEr
- **详细文档**: [README](audio/audiocaps/README.md)

### 10. [MultiHop-RAG](rag/multihop_rag/) (多跳推理)
- **任务**: 多文档检索+推理问答
- **来源**: 英语新闻
- **特点**: 每问题需 2-4 个文档
- **评测指标**: R@K, EM, F1
- **详细文档**: [README](rag/multihop_rag/README.md)

---

## 附加数据集

### 11. [WildChat 10K](conversation/wildchat_10k/) (AI对话)
- **任务**: AI 创意写作对话检索
- **来源**: ChatGPT 等 LLM 的真实用户对话
- **特点**: 包含创意性、质量评分等标注
- **评测指标**: 对话检索、质量评估
- **详细文档**: [README](conversation/wildchat_10k/README.md)

### 12. [Discord Chat](conversation/discord_chat/) (社交聊天)
- **任务**: 即时通讯消息检索
- **来源**: Discord 平台聊天记录
- **特点**: 非正式对话风格
- **评测指标**: 消息检索、对话分析
- **详细文档**: [README](conversation/discord_chat/README.md)

### 13. [ELI5 Reddit](conversation/eli5_reddit/) (问答社区)
- **任务**: 长文本开放域问答
- **来源**: Reddit r/explainlikeimfive
- **特点**: 通俗易懂的知识解释，含投票分数
- **评测指标**: R@K, ROUGE, BERTScore
- **详细文档**: [README](conversation/eli5_reddit/README.md)

### 14. [GitHub Issues](code/github_issues/) (代码Issue)
- **任务**: Issue/PR 检索与分类
- **来源**: GitHub 开源项目
- **特点**: 完整的 Issue 元数据和评论
- **评测指标**: R@K, 分类 F1
- **详细文档**: [README](code/github_issues/README.md)

### 15. [Enron Email](document/enron_mini/) (企业邮件)
- **任务**: 邮件检索、垃圾邮件分类
- **来源**: Enron 公司邮件语料库
- **特点**: 经典文本分类基准，含 ham/spam 标签
- **评测指标**: Accuracy, F1, AUC-ROC
- **详细文档**: [README](document/enron_mini/README.md)

## 许可证

各数据集的许可证请参见各自目录下的 README.md 文件。大部分数据集采用 CC BY 4.0 或类似许可。

## Milvus 向量搜索 Demo

所有数据集都提供了基于 Milvus Lite + OpenAI Embedding 的搜索 Demo。

详见：[demos/README.md](../demos/README.md)

运行示例：
```bash
export OPENAI_API_KEY="your-api-key"
python demos/demo_squad2.py
```

## 更新日期

2026-01-30
