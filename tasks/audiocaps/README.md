# AudioCaps 音频检索

## 任务描述

AudioCaps 是一个音频描述数据集，用于评估音频与文本之间的检索能力。任务包括：
1. **音频到文本检索 (A2T)**: 给定音频，检索最相关的文本描述
2. **文本到音频检索 (T2A)**: 给定文本描述，检索最匹配的音频

## 数据集

- **来源**: `AudioLLMs/audiocaps_test`
- **规模**: ~50,000 音频-文本对
- **语言**: 英语
- **音频来源**: YouTube 视频

## 任务目标

构建音频-文本嵌入模型，能够：
1. 将音频和文本编码到同一向量空间
2. 通过向量相似度进行跨模态检索

## 评估指标

| 指标 | 说明 |
|------|------|
| **R@1** | 正确结果出现在第 1 位的比例 |
| **R@5** | 正确结果出现在前 5 位的比例 |
| **R@10** | 正确结果出现在前 10 位的比例 |
| **mAP** | 平均精度均值 |

## 提交格式

```json
{
  "model_name": "your-model-name",
  "audio_to_text": {
    "audio_id_1": ["text_id_1", "text_id_3", "text_id_2"],
    "audio_id_2": ["text_id_5", "text_id_1", "text_id_8"]
  },
  "text_to_audio": {
    "text_id_1": ["audio_id_1", "audio_id_3", "audio_id_2"],
    "text_id_2": ["audio_id_5", "audio_id_1", "audio_id_4"]
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
  "task": "audiocaps",
  "a2t_r@1": 35.2,
  "a2t_r@5": 62.1,
  "a2t_r@10": 75.4,
  "t2a_r@1": 28.3,
  "t2a_r@5": 55.8,
  "t2a_r@10": 68.2,
  "num_samples": 975,
  "timestamp": "2024-01-30T12:00:00"
}
```

## 参考资料

- [AudioCaps 论文](https://arxiv.org/abs/1706.10006)
- [CLAP 论文](https://arxiv.org/abs/2211.06687)
