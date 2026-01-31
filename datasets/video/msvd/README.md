# MSVD 数据集

## 概述

**MSVD**（Microsoft Video Description）是微软发布的视频描述数据集，是视频理解领域的经典小型基准。

- **发布机构**: Microsoft Research
- **论文**: "Collecting Highly Parallel Data for Paraphrase Evaluation" (ACL 2011)
- **HuggingFace**: `friedrichor/MSVD`
- **许可证**: 研究用途
- **语言**: 英语（原始）、多语言（扩展）

## 数据集特点

- **小而精**: 仅约 2,000 个视频，适合快速实验
- **多描述**: 每个视频约 40 条描述（众包标注）
- **短视频**: 平均时长约 10 秒
- **YouTube 来源**: 视频来自 YouTube

## 数据集规模

| 子集 | 视频数 | 描述数 | 说明 |
|------|--------|--------|------|
| train | 1,200 | ~48,000 | 训练集 |
| validation | 100 | ~4,000 | 验证集 |
| test | 670 | ~26,800 | 测试集 |

**总计**: ~1,970 个视频，~80,000 条描述

## 数据字段说明

| 字段名 | 类型 | 说明 |
|--------|------|------|
| `video_id` | string | YouTube 视频 ID |
| `video` | bytes/path | 视频数据或路径 |
| `caption` | string | 视频描述文本 |
| `start_time` | float | 视频片段起始时间（秒） |
| `end_time` | float | 视频片段结束时间（秒） |

## 数据示例

```python
{
    "video_id": "mv89psg6zh4_33_46",
    "caption": "A panda is eating bamboo.",
    "start_time": 33.0,
    "end_time": 46.0
}
```

同一视频的多条描述示例：

```
Video: mv89psg6zh4_33_46
- "A panda is eating bamboo."
- "A cute panda bear eats bamboo leaves."
- "The panda is chewing on some bamboo."
- "A giant panda sits and eats bamboo."
- ...（约 40 条）
```

## 视频内容类型

MSVD 涵盖多种日常活动：
- 动物行为
- 烹饪活动
- 体育运动
- 人际互动
- 音乐表演
- 手工制作

## 评测任务

### 1. 视频描述生成 (Video Captioning)
- **输入**: 视频
- **输出**: 描述文本
- **指标**: BLEU, METEOR, CIDEr, ROUGE-L

### 2. 视频文本检索 (Video-Text Retrieval)
- **输入**: 文本查询或视频
- **输出**: 最相关的视频或文本
- **指标**: R@1, R@5, R@10, MedR

## 评测指标说明

| 指标 | 说明 | 适用任务 |
|------|------|----------|
| BLEU | n-gram 精确度 | 描述生成 |
| METEOR | 考虑同义词的匹配 | 描述生成 |
| CIDEr | 基于 TF-IDF 的共识度量 | 描述生成 |
| R@K | Top-K 召回率 | 检索 |
| MedR | 中位排名 | 检索 |

## 使用方法

```python
from datasets import load_dataset

# 加载数据集
dataset = load_dataset("friedrichor/MSVD")

# 查看示例
example = dataset["train"][0]
print("Video ID:", example["video_id"])
print("Caption:", example["caption"])

# 获取同一视频的所有描述
video_id = example["video_id"]
same_video = dataset["train"].filter(
    lambda x: x["video_id"] == video_id
)
print(f"Found {len(same_video)} captions for this video")
```

## 视频下载说明

由于版权原因，部分 HuggingFace 版本可能只包含视频 ID 和描述，需要自行下载视频：

1. 使用 `yt-dlp` 下载：
```bash
yt-dlp -o "videos/%(id)s.%(ext)s" "https://youtube.com/watch?v=VIDEO_ID"
```

2. 使用预提取的视频特征（推荐）

## 与其他视频数据集比较

| 数据集 | 视频数 | 描述数/视频 | 特点 |
|--------|--------|-------------|------|
| **MSVD** | 2K | ~40 | 小巧经典 |
| MSR-VTT | 10K | 20 | 更大规模 |
| ActivityNet | 20K | ~5 | 长视频 |
| YouCook2 | 2K | 多步骤 | 烹饪领域 |

## 应用场景

1. **视频理解模型快速验证**: 数据量小，迭代快
2. **视频描述生成**: 经典评测基准
3. **视频问答**: 结合描述进行 QA
4. **跨模态预训练**: 作为下游任务评测

## 参考链接

- 原始数据: https://www.cs.utexas.edu/users/ml/clamp/videoDescription/
- 论文: https://aclanthology.org/P11-1020/
- HuggingFace: https://huggingface.co/datasets/friedrichor/MSVD
