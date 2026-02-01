# 数据集

本目录存放下载的数据集文件。每个数据集对应 `tasks/` 目录中的一个或多个任务。

## 目录结构

```
datasets/
├── text/           # 文本数据集
│   └── squad2/     # SQuAD 2.0
├── chinese/        # 中文数据集
│   └── cmrc2018/   # CMRC 2018
├── multimodal/     # 多模态数据集
│   ├── chartqa/    # ChartQA
│   └── coco_karpathy/  # COCO Karpathy
├── audio/          # 音频数据集
│   └── audiocaps/  # AudioCaps
├── video/          # 视频数据集
│   └── msvd/       # MSVD
├── finance/        # 金融数据集
│   └── finqa/      # FinQA
├── table/          # 表格数据集
│   └── spider/     # Spider
├── document/       # 文档数据集
│   ├── cuad/       # CUAD
│   └── enron_mini/ # Enron
├── rag/            # RAG 数据集
│   └── multihop_rag/  # MultiHop-RAG
├── conversation/   # 对话数据集
│   ├── wildchat_10k/  # WildChat
│   ├── discord_chat/  # Discord
│   └── eli5_reddit/   # ELI5
└── code/           # 代码数据集
    └── github_issues/ # GitHub Issues
```

## 数据下载

使用提供的脚本下载数据集：

```bash
python scripts/download_datasets.py
```

## 大文件说明

为节省存储空间，以下类型的大文件通过 `.gitignore` 排除：
- `.parquet` - 数据表文件
- `.arrow` - Arrow 格式文件
- `.h5`, `.hdf5` - HDF5 文件
- `.pkl`, `.pickle` - Pickle 序列化文件
- `.bin`, `.pt`, `.onnx`, `.safetensors` - 模型权重文件

这些文件需要通过下载脚本获取。

## 各数据集详情

每个数据集子目录包含：
- `README.md` - 数据集详细说明
- `metadata.json` - 数据集元信息
- 数据文件（需下载）
