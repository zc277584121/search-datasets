# MSVD 视频检索

## 任务描述

MSVD (Microsoft Research Video Description) 是一个视频描述数据集，用于评估视频与文本之间的检索能力。

## 数据集信息

- **来源**: `friedrichor/MSVD`
- **评测集**: 500 条
- **语言**: 英语

## 数据格式

### queries.json 字段说明

```json
{
  "task": "msvd",
  "total": 500,
  "queries": [
    {
      "id": "0",
      "video_id": "vid1"
    }
  ]
}
```

| 字段 | 类型 | 说明 |
|------|------|------|
| `id` | string | 查询唯一标识符 |
| `video_id` | string | MSVD 视频 ID |

### 加载视频

```python
from datasets import load_dataset

dataset = load_dataset('friedrichor/MSVD', split='test')
```

## 使用流程

### 1. 加载评测数据

```python
import json

with open("queries.json", "r") as f:
    data = json.load(f)

video_embeddings = {}
for query in data["queries"]:
    qid = query["id"]
    video_id = query["video_id"]

    # 加载视频并编码
    video = load_video(video_id)
    video_embeddings[qid] = your_model.encode_video(video)
```

### 2. 生成预测结果

```json
{
  "model_name": "your-model-name",
  "video_to_text": {
    "0": ["0_0", "0_1", "1_0"],
    "1": ["1_0", "1_1", "0_0"]
  },
  "text_to_video": {
    "0_0": ["0", "1", "2"],
    "1_0": ["1", "0", "3"]
  }
}
```

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
| **MdR** | 中位排名 (Median Rank) |

## 输出示例

```json
{
  "task": "msvd",
  "model_name": "your-model",
  "v2t_r@1": 45.2,
  "v2t_r@5": 72.1,
  "t2v_r@1": 32.3,
  "t2v_r@5": 61.8,
  "num_queries": 500
}
```

## 参考资料

- [MSVD 数据集](https://www.cs.utexas.edu/users/ml/clamp/videoDescription/)
