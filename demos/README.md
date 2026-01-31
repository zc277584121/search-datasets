# Milvus 向量搜索 Demo

本目录包含使用 Milvus Lite 的搜索 Demo 脚本，支持文本和多模态检索。

## 环境要求

```bash
# 文本检索需要设置 OpenAI API Key
export OPENAI_API_KEY="your-api-key"

# 多模态检索使用本地 GPU 运行 CLIP/CLAP 模型
# 需要 CUDA 支持的 GPU

# 依赖已在项目中安装
# - pymilvus, milvus-lite
# - openai (text-embedding-3-small)
# - transformers, torch (CLIP, CLAP 模型)
```

## Demo 列表

### 文本检索 (OpenAI Embedding)

| Demo 脚本 | 数据集 | 场景 | 说明 |
|-----------|--------|------|------|
| `demo_squad2.py` | SQuAD 2.0 | 文本问答 | 问题 → 相关段落检索 |
| `demo_cmrc2018.py` | CMRC 2018 | 中文阅读理解 | 中文问题 → 中文段落检索 |
| `demo_multihop_rag.py` | MultiHop-RAG | RAG 文档检索 | 查询 → 新闻文档检索 |
| `demo_eli5_reddit.py` | ELI5 Reddit | 问答社区 | 问题 → 相似问答检索 |
| `demo_wildchat.py` | WildChat 10K | AI 对话 | 主题 → 创意写作对话检索 |
| `demo_github_issues.py` | GitHub Issues | Issue 检索 | 问题描述 → 相似 Issue 检索 |
| `demo_enron_email.py` | Enron Email | 企业邮件 | 关键词 → 相关邮件检索 |
| `demo_discord_chat.py` | Discord Chat | 聊天检索 | 话题 → 相关聊天检索 |
| `demo_spider_sql.py` | Spider | NL2SQL | 问题 → 相似 SQL 检索 |
| `demo_coco_captions.py` | COCO Karpathy | 图文检索(文本) | 图像描述 → 相似描述检索 |
| `demo_msvd_video.py` | MSVD | 视频描述(文本) | 视频描述 → 相似视频检索 |
| `demo_audiocaps.py` | AudioCaps | 音频描述(文本) | 声音描述 → 相似音频检索 |

### 多模态检索 (本地 GPU)

| Demo 脚本 | 数据集 | 模型 | 说明 |
|-----------|--------|------|------|
| `demo_coco_multimodal.py` | COCO Karpathy | CLIP | 文本↔图像 双向检索 |
| `demo_chartqa_multimodal.py` | ChartQA | CLIP | 文本↔图表 双向检索 |
| `demo_audiocaps_multimodal.py` | AudioCaps | CLAP | 文本↔音频 双向检索 |

## 使用方法

```bash
# 运行文本检索 demo（需要 OPENAI_API_KEY）
python demos/demo_squad2.py

# 运行多模态检索 demo（需要 GPU）
python demos/demo_coco_multimodal.py      # 图像检索
python demos/demo_chartqa_multimodal.py   # 图表检索
python demos/demo_audiocaps_multimodal.py # 音频检索
```

## 技术实现

### 文本 Embedding
- **模型**: OpenAI `text-embedding-3-small`
- **维度**: 1536
- **特点**: 高性能、低成本、支持多语言

### 多模态 Embedding
- **图像/图表**: CLIP (`openai/clip-vit-base-patch32`)，维度 512
- **音频**: CLAP (`laion/larger_clap_music_and_speech`)，维度 512
- **运行**: 本地 GPU (CUDA)

### Milvus 配置
- **模式**: Milvus Lite (本地文件存储)
- **接口**: MilvusClient (非 ORM)
- **URI**: `*.db` 文件
- **度量**: Cosine 相似度

### 数据处理
- 每个 demo 索引 100-500 条数据用于演示
- 多模态 demo 直接处理原始媒体数据（图像/音频）
- 元数据存储在 Milvus collection 中

## 文件结构

```
demos/
├── README.md                    # 本文档
├── utils.py                     # 共享工具模块（OpenAI embedding）
├── run_all_demos.py             # 批量运行脚本
│
├── # 文本检索 Demo (OpenAI)
├── demo_squad2.py               # SQuAD 2.0 问答
├── demo_cmrc2018.py             # CMRC 2018 中文
├── demo_multihop_rag.py         # MultiHop-RAG
├── demo_eli5_reddit.py          # ELI5 Reddit
├── demo_wildchat.py             # WildChat AI对话
├── demo_github_issues.py        # GitHub Issues
├── demo_enron_email.py          # Enron Email
├── demo_discord_chat.py         # Discord Chat
├── demo_spider_sql.py           # Spider NL2SQL
├── demo_coco_captions.py        # COCO 文本检索
├── demo_msvd_video.py           # MSVD 文本检索
├── demo_audiocaps.py            # AudioCaps 文本检索
│
├── # 多模态检索 Demo (本地 GPU)
├── demo_coco_multimodal.py      # COCO 图像检索 (CLIP)
├── demo_chartqa_multimodal.py   # ChartQA 图表检索 (CLIP)
├── demo_audiocaps_multimodal.py # AudioCaps 音频检索 (CLAP)
│
└── *.db                         # Milvus Lite 数据库（运行后生成）
```

## 扩展建议

1. **增加数据量**: 修改 `SAMPLE_SIZE` 参数索引更多数据
2. **混合检索**: 结合关键词过滤和向量搜索
3. **重排序**: 使用 reranker 模型对结果重排序
4. **多模态**: 对于图像/视频数据集，可添加 CLIP 等视觉编码器

## 注意事项

- 首次运行需要生成 embedding，会消耗 OpenAI API 额度
- 每个 demo 会创建一个 `.db` 文件用于存储索引
- 交互模式输入 `quit` 或 `q` 退出
