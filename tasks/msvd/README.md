# MSVD 视频检索

## 任务描述

MSVD (Microsoft Research Video Description) 是一个视频描述数据集，用于评估视频与文本之间的检索能力。任务包括：
1. **视频到文本检索 (V2T)**: 给定视频，检索最相关的文本描述
2. **文本到视频检索 (T2V)**: 给定文本描述，检索最匹配的视频

## 数据集

- **来源**: `friedrichor/MSVD`
- **规模**: ~2,000 视频片段，约 80,000 描述
- **语言**: 英语
- **视频时长**: 1-62 秒

## 任务目标

构建视频-文本嵌入模型，能够：
1. 将视频和文本编码到同一向量空间
2. 通过向量相似度进行跨模态检索

## 评估指标

| 指标 | 说明 |
|------|------|
| **R@1** | 正确结果出现在第 1 位的比例 |
| **R@5** | 正确结果出现在前 5 位的比例 |
| **R@10** | 正确结果出现在前 10 位的比例 |
| **MdR** | 中位排名 (Median Rank) |

## 提交格式

```json
{
  "model_name": "your-model-name",
  "video_to_text": {
    "video_id_1": ["text_id_1", "text_id_3", "text_id_2"],
    "video_id_2": ["text_id_5", "text_id_1", "text_id_8"]
  },
  "text_to_video": {
    "text_id_1": ["video_id_1", "video_id_3", "video_id_2"],
    "text_id_2": ["video_id_5", "video_id_1", "video_id_4"]
  }
}
```

## 运行评估

```bash
python eval.py --submission predictions.json
```

## 输出示例

```json
{
  "task": "msvd",
  "v2t_r@1": 45.2,
  "v2t_r@5": 72.1,
  "v2t_r@10": 83.4,
  "t2v_r@1": 32.3,
  "t2v_r@5": 61.8,
  "t2v_r@10": 74.2,
  "num_videos": 670,
  "timestamp": "2024-01-30T12:00:00"
}
```

## 参考资料

- [MSVD 数据集](https://www.cs.utexas.edu/users/ml/clamp/videoDescription/)
- [VideoCLIP 论文](https://arxiv.org/abs/2109.14084)
