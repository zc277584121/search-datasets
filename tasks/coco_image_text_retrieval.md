# 任务：COCO 图文检索

## 任务描述

构建一个跨模态检索系统，能够：
1. **图像 → 文本**：给定一张图片，找到最相关的文本描述
2. **文本 → 图像**：给定一条文本描述，找到最相关的图片

## 数据集信息

| 属性 | 值 |
|------|-----|
| **名称** | COCO Karpathy 5K 测试集 |
| **来源** | `nlphuji/mscoco_2014_5k_test_image_text_retrieval` |
| **语言** | 英语 |
| **许可证** | CC BY 4.0 |
| **规模** | 5,000 张图片，25,000 条描述（每张图 5 条） |

### 数据字段

| 字段 | 类型 | 描述 |
|------|------|------|
| `image` | Image | 图片 |
| `image_id` | int | COCO 图片 ID |
| `caption` | string | 文本描述 |
| `caption_id` | int | 描述 ID |

### 检索设置

- **图像→文本**：用 1 张图片查询，从 25,000 条描述中检索
- **文本→图像**：用 1 条描述查询，从 5,000 张图片中检索
- 每张图片恰好有 5 条标准描述

## 评估目标

达到：
- **图像→文本 R@1 ≥ 50%**
- **文本→图像 R@1 ≥ 35%**
- **平均召回率（所有 R@K 的平均）≥ 70%**

## 评估方法

### Recall@K 指标

```python
import numpy as np

def compute_retrieval_metrics(similarities, ground_truth_indices):
    """
    计算检索指标

    参数:
        similarities: (N, M) 相似度矩阵
        ground_truth_indices: 每个查询的标准答案索引列表

    返回:
        包含 R@1, R@5, R@10, MeanR, MedianR 的字典
    """
    N = similarities.shape[0]
    ranks = []

    for i in range(N):
        # 按相似度降序排列
        sorted_indices = np.argsort(-similarities[i])

        # 找到第一个标准答案的排名
        gt = set(ground_truth_indices[i])
        for rank, idx in enumerate(sorted_indices):
            if idx in gt:
                ranks.append(rank + 1)  # 1-indexed
                break

    ranks = np.array(ranks)

    return {
        'R@1': 100.0 * np.mean(ranks <= 1),
        'R@5': 100.0 * np.mean(ranks <= 5),
        'R@10': 100.0 * np.mean(ranks <= 10),
        'MeanR': np.mean(ranks),
        'MedianR': np.median(ranks)
    }

def evaluate_retrieval(image_embeddings, text_embeddings, image_ids, caption_image_ids):
    """
    评估双向检索

    参数:
        image_embeddings: (N_images, D) 图像特征矩阵
        text_embeddings: (N_texts, D) 文本特征矩阵
        image_ids: 每张图片的 ID 列表
        caption_image_ids: 每条描述对应的图片 ID
    """
    # 计算相似度矩阵
    similarities = np.dot(text_embeddings, image_embeddings.T)  # (N_texts, N_images)

    # 文本 → 图像检索
    t2i_gt = []
    for caption_img_id in caption_image_ids:
        gt_indices = [i for i, img_id in enumerate(image_ids) if img_id == caption_img_id]
        t2i_gt.append(gt_indices)

    t2i_metrics = compute_retrieval_metrics(similarities, t2i_gt)

    # 图像 → 文本检索
    i2t_gt = []
    for img_id in image_ids:
        gt_indices = [i for i, cap_img_id in enumerate(caption_image_ids) if cap_img_id == img_id]
        i2t_gt.append(gt_indices)

    i2t_metrics = compute_retrieval_metrics(similarities.T, i2t_gt)

    return {
        'text_to_image': t2i_metrics,
        'image_to_text': i2t_metrics
    }
```

## 建议方案

1. **基于 CLIP**：使用 CLIP 或 OpenCLIP 获取联合图文向量

2. **BLIP/BLIP-2**：使用最先进的视觉语言模型

3. **微调模型**：在 COCO 训练集上微调

4. **模型集成**：组合多个模型以获得更好的性能

## 核心挑战

- 理解细粒度的视觉细节
- 处理每张图片的多个有效描述
- 扩展到大规模检索池
- 跨模态语义对齐

## 基线性能

| 模型 | I2T R@1 | T2I R@1 |
|------|---------|---------|
| CLIP ViT-B/32 | 50.1 | 30.4 |
| CLIP ViT-L/14 | 58.4 | 37.8 |
| BLIP | 82.4 | 65.1 |
| BLIP-2 | 85.8 | 67.5 |

## 参考资料

- 原始数据集: https://cocodataset.org/
- Karpathy 划分: [Deep Visual-Semantic Alignments](https://cs.stanford.edu/people/karpathy/deepimagesent/)
