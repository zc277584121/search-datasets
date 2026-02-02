# COCO 图文检索

## 任务描述

MS COCO 图文检索任务评估模型在图像-文本双向检索上的能力。

## 数据集信息

- **来源**: `nlphuji/mscoco_2014_5k_test`
- **评测集**: 500 条
- **语言**: 英语

## 数据格式

### queries.json 字段说明

| 字段 | 类型 | 说明 |
|------|------|------|
| `id` | string | 样本唯一标识符 |
| `image_id` | int | 图片 ID |

### 预测结果格式

```json
{
  "model_name": "your-model-name",
  "image_to_text": {"image_id": ["caption_id_1", "caption_id_2"]},
  "text_to_image": {"caption_id": ["image_id_1", "image_id_2"]}
}
```

## 快速开始

1. 打开 `run_demo.py`，找到 `# TODO` 注释，替换为你的模型代码
2. 运行：
   ```bash
   python run_demo.py
   ```

## 评估指标

| 指标 | 说明 |
|------|------|
| **R@1/5/10** | 召回率 |
| **Mean Rank** | 平均排名 |

## 参考资料

- [MS COCO](https://cocodataset.org/)
