# COCO 图文检索

## 任务描述

COCO 图文检索任务评估模型在图像与文本之间进行双向检索的能力。包含两个子任务：
1. **图像到文本检索 (I2T)**: 给定图像，检索最相关的文本描述
2. **文本到图像检索 (T2I)**: 给定文本描述，检索最匹配的图像

## 数据集

- **来源**: `nlphuji/mscoco_2014_5k_test`
- **规模**: 5,000 张测试图像，每张图像有 5 个描述
- **语言**: 英语

## 任务目标

构建图文嵌入模型，能够：
1. 将图像和文本编码到同一向量空间
2. 通过向量相似度进行跨模态检索

## 评估指标

| 指标 | 说明 |
|------|------|
| **R@1** | 正确结果出现在第 1 位的比例 |
| **R@5** | 正确结果出现在前 5 位的比例 |
| **R@10** | 正确结果出现在前 10 位的比例 |
| **Mean Rank** | 正确结果的平均排名 |

分别报告 I2T 和 T2I 两个方向的指标。

## 提交格式

```json
{
  "model_name": "your-model-name",
  "image_to_text": {
    "image_id_1": ["text_id_1", "text_id_2", "text_id_3"],
    "image_id_2": ["text_id_5", "text_id_1", "text_id_8"]
  },
  "text_to_image": {
    "text_id_1": ["image_id_1", "image_id_3", "image_id_2"],
    "text_id_2": ["image_id_5", "image_id_1", "image_id_4"]
  }
}
```

**说明**:
- 每个查询返回按相关性排序的结果列表
- 建议返回 Top-10 或 Top-100 结果

## 运行评估

```bash
python eval.py --submission predictions.json
```

## 输出示例

```json
{
  "task": "coco",
  "i2t_r@1": 65.2,
  "i2t_r@5": 87.1,
  "i2t_r@10": 93.4,
  "t2i_r@1": 48.3,
  "t2i_r@5": 76.8,
  "t2i_r@10": 86.2,
  "num_images": 5000,
  "num_texts": 25000,
  "timestamp": "2024-01-30T12:00:00"
}
```

## 参考资料

- [MS COCO 数据集](https://cocodataset.org/)
- [CLIP 论文](https://arxiv.org/abs/2103.00020)
