# COCO 图文检索提交格式

## 任务说明

双向图文检索：
1. 图像 → 文本：给定图片，检索最相关的描述
2. 文本 → 图像：给定描述，检索最相关的图片

## 输入

评估数据 `datasets/multimodal/coco_karpathy/`：
- 5,000 张图片，25,000 条描述（每张图 5 条）
- `image_id`: 图片 ID
- `caption`: 描述文本
- `caption_id`: 描述 ID

## 输出格式

提交文件 `predictions.json`，格式如下：

```json
{
  "model_name": "你的模型名称",
  "image_to_text": {
    "391895": ["caption_id_1", "caption_id_2", ...],
    "522418": ["caption_id_3", "caption_id_4", ...],
    ...
  },
  "text_to_image": {
    "caption_id_1": ["image_id_1", "image_id_2", ...],
    "caption_id_2": ["image_id_3", "image_id_4", ...],
    ...
  }
}
```

### 字段说明

| 字段 | 类型 | 说明 |
|------|------|------|
| `image_to_text` | object | 图片ID到检索的描述ID列表（按相关性排序） |
| `text_to_image` | object | 描述ID到检索的图片ID列表（按相关性排序） |

### 注意事项

- 每个列表按相关性**降序**排列
- 至少返回 top-10 结果用于评估 R@10
- ID 类型与数据集保持一致

## 评估指标

- **R@1, R@5, R@10**: 召回率
- **MRR**: 平均倒数排名

## 运行评估

```bash
python eval/run_eval.py --task coco --submission submissions/coco/predictions.json
```
