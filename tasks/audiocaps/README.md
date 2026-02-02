# AudioCaps 音频检索

## 任务描述

AudioCaps 是一个音频描述数据集，用于评估音频与文本之间的检索能力。

## 数据集信息

- **来源**: `AudioLLMs/audiocaps_test`
- **评测集**: 500 条
- **语言**: 英语

## 数据格式

### queries.json 字段说明

```json
{
  "task": "audiocaps",
  "total": 500,
  "queries": [
    {
      "id": "0",
      "audiocap_id": "audiocap_0",
      "youtube_id": "vid0_7fmOlUlwoNg",
      "start_time": 0
    }
  ]
}
```

| 字段 | 类型 | 说明 |
|------|------|------|
| `id` | string | 查询唯一标识符 |
| `audiocap_id` | string | AudioCaps 原始 ID |
| `youtube_id` | string | YouTube 视频 ID |
| `start_time` | int | 音频起始时间（秒） |

### 加载音频

```python
from datasets import load_dataset

# 从 HuggingFace 加载
dataset = load_dataset('AudioLLMs/audiocaps_test', split='test')
```

## 使用流程

### 1. 加载评测数据

```python
import json

with open("queries.json", "r") as f:
    data = json.load(f)

# 构建音频嵌入
audio_embeddings = {}
for query in data["queries"]:
    qid = query["id"]
    # 加载音频文件
    audio = load_audio(query["youtube_id"], query["start_time"])
    audio_embeddings[qid] = your_model.encode_audio(audio)
```

### 2. 生成预测结果

```json
{
  "model_name": "your-model-name",
  "audio_to_text": {
    "0": ["0_caption", "1_caption", "2_caption"],
    "1": ["1_caption", "0_caption", "3_caption"]
  },
  "text_to_audio": {
    "0_caption": ["0", "1", "2"],
    "1_caption": ["1", "0", "3"]
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
| **mAP** | 平均精度均值 |

## 输出示例

```json
{
  "task": "audiocaps",
  "model_name": "your-model",
  "a2t_r@1": 35.2,
  "a2t_r@5": 62.1,
  "t2a_r@1": 28.3,
  "t2a_r@5": 55.8,
  "num_queries": 500
}
```

## 参考资料

- [AudioCaps 论文](https://arxiv.org/abs/1706.10006)
- [CLAP 论文](https://arxiv.org/abs/2211.06687)
