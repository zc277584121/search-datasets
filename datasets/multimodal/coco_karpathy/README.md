# COCO Karpathy 5K 数据集

## 概述

**COCO Karpathy 5K** 是 MS COCO 数据集的 Karpathy 划分版本的测试集，是**图文检索**（Image-Text Retrieval）领域最标准的评测基准。

- **原始数据集**: Microsoft COCO (Common Objects in Context)
- **划分方式**: Karpathy & Li (2015)
- **HuggingFace**: `nlphuji/mscoco_2014_5k_test_image_text_retrieval`
- **许可证**: CC BY 4.0
- **语言**: 英语

## 数据集特点

- **标准评测集**: 图文检索研究的事实标准
- **高质量描述**: 每张图片有 5 个人工标注的描述
- **5K 测试集**: 专门用于评测的 5000 张图片子集
- **双向检索**: 支持图搜文和文搜图两个任务

## 数据集规模

### Karpathy 划分

| 子集 | 图片数 | 描述数 | 说明 |
|------|--------|--------|------|
| train | 113,287 | 566,435 | 训练集 |
| validation | 5,000 | 25,000 | 验证集 |
| test | 5,000 | 25,000 | 测试集（本数据集） |

**本数据集**: 5,000 张图片 + 25,000 条描述

## 数据字段说明

| 字段名 | 类型 | 说明 |
|--------|------|------|
| `image` | Image | 图片数据 |
| `image_id` | int | COCO 图片 ID |
| `caption` | string | 图片描述文本 |
| `caption_id` | int | 描述 ID |

## 数据示例

```python
{
    "image": <PIL.Image>,  # 一张图片
    "image_id": 391895,
    "caption": "A man with a red helmet on a small moped on a dirt road.",
    "caption_id": 12345
}
```

每张图片对应 5 条不同的描述：

```
Image ID: 391895
- Caption 1: "A man with a red helmet on a small moped on a dirt road."
- Caption 2: "Man riding a motor bike on a dirt road on the countryside."
- Caption 3: "A person riding a motorcycle on a dirt path."
- Caption 4: "A man in a red shirt is on a moped on a dirt road."
- Caption 5: "A man is riding a motorcycle down a dirt road."
```

## 评测任务

### 1. 图像到文本检索 (Image-to-Text Retrieval)
- **输入**: 一张图片
- **输出**: 从 25,000 条描述中检索最相关的
- **指标**: R@1, R@5, R@10

### 2. 文本到图像检索 (Text-to-Image Retrieval)
- **输入**: 一条描述
- **输出**: 从 5,000 张图片中检索最相关的
- **指标**: R@1, R@5, R@10

## 评测指标

| 指标 | 说明 |
|------|------|
| R@1 | 正确结果在 Top-1 的比例 |
| R@5 | 正确结果在 Top-5 的比例 |
| R@10 | 正确结果在 Top-10 的比例 |
| Mean Rank | 正确结果排名的平均值 |
| Median Rank | 正确结果排名的中位数 |

## 使用方法

```python
from datasets import load_dataset

# 加载数据集
dataset = load_dataset("nlphuji/mscoco_2014_5k_test_image_text_retrieval")

# 查看示例
example = dataset["test"][0]
print("Caption:", example["caption"])

# 显示图片
example["image"].show()

# 获取同一张图片的所有描述
image_id = example["image_id"]
same_image = dataset["test"].filter(
    lambda x: x["image_id"] == image_id
)
for item in same_image:
    print(item["caption"])
```

## 与其他划分的区别

| 划分方式 | 测试集大小 | 说明 |
|----------|------------|------|
| Karpathy | 5,000 | 最常用，本数据集 |
| COCO 2014 官方 | 40,504 | 官方验证集 |
| 1K Test | 1,000 | 部分论文使用 |

## 图片内容分布

COCO 数据集涵盖 80 类物体，图片内容包括：
- 人物活动（运动、烹饪、工作等）
- 动物
- 交通工具
- 室内/室外场景
- 食物
- 家具和电器

## 应用场景

1. **多模态检索模型评测**: CLIP、BLIP、ALIGN 等
2. **跨模态对齐研究**: 学习图像和文本的联合表示
3. **图文匹配**: 判断图片和描述是否匹配
4. **图像描述生成**: 为图片生成自然语言描述

## 经典模型在该数据集上的表现

| 模型 | I2T R@1 | T2I R@1 | 年份 |
|------|---------|---------|------|
| CLIP (ViT-L/14) | 58.4 | 37.8 | 2021 |
| BLIP | 82.4 | 65.1 | 2022 |
| BLIP-2 | 85.8 | 67.5 | 2023 |

## 相关数据集

| 数据集 | 特点 |
|--------|------|
| Flickr30K | 31K 图片，另一经典基准 |
| Visual Genome | 更详细的区域描述 |
| CC3M/CC12M | 大规模网络图文对 |

## 参考链接

- MS COCO 官网: https://cocodataset.org/
- Karpathy 划分: https://cs.stanford.edu/people/karpathy/deepimagesent/
- 论文: "Deep Visual-Semantic Alignments for Generating Image Descriptions" (CVPR 2015)
