# COCO 图文检索

## 任务描述

COCO 图文检索任务评估模型在图像与文本之间进行双向检索的能力。

## 数据集信息

- **来源**: MS COCO 2014
- **评测集**: 500 条
- **语言**: 英语

## 数据格式

### queries.json 字段说明

```json
{
  "task": "coco",
  "total": 500,
  "queries": [
    {
      "id": "391895",
      "image_id": 391895
    }
  ]
}
```

| 字段 | 类型 | 说明 |
|------|------|------|
| `id` | string | 查询唯一标识符 |
| `image_id` | int | COCO 图像 ID |

### 加载图像

需要从 MS COCO 数据集下载图像：
```python
# 下载 COCO 2014 Val images
# http://images.cocodataset.org/zips/val2014.zip
```

## 使用流程

### 1. 加载评测数据

```python
import json
from PIL import Image

with open("queries.json", "r") as f:
    data = json.load(f)

# 构建图像和文本的嵌入
image_embeddings = {}
for query in data["queries"]:
    image_id = query["image_id"]
    image_path = f"val2014/COCO_val2014_{image_id:012d}.jpg"
    image = Image.open(image_path)

    # 用你的模型编码图像
    image_embeddings[query["id"]] = your_model.encode_image(image)
```

### 2. 生成预测结果

```json
{
  "model_name": "your-model-name",
  "image_to_text": {
    "391895": ["391895_0", "391895_2", "522418_1"],
    "522418": ["522418_0", "522418_1", "391895_2"]
  },
  "text_to_image": {
    "391895_0": ["391895", "522418", "318219"],
    "522418_0": ["522418", "391895", "318219"]
  }
}
```

**说明**:
- `image_to_text`: 图像 ID → 排序后的文本 ID 列表
- `text_to_image`: 文本 ID → 排序后的图像 ID 列表

### 3. 运行评估

```bash
python eval.py --submission predictions.json
```

## 评估指标

| 指标 | 说明 |
|------|------|
| **R@1** | 正确结果出现在第 1 位的比例 |
| **R@5** | 正确结果出现在前 5 位的比例 |
| **R@10** | 正确结果出现在前 10 位的比例 |

## 输出示例

```json
{
  "task": "coco",
  "model_name": "your-model",
  "i2t_r@1": 65.2,
  "i2t_r@5": 87.1,
  "t2i_r@1": 48.3,
  "t2i_r@5": 76.8,
  "num_queries": 500
}
```

## 参考资料

- [MS COCO 数据集](https://cocodataset.org/)
- [CLIP 论文](https://arxiv.org/abs/2103.00020)
