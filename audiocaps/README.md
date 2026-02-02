# AudioCaps 音频文本检索

## 任务描述

AudioCaps 评估模型在音频-文本双向检索任务上的能力。

## 数据集信息

- **来源**: `AudioLLMs/audiocaps_test`
- **评测集**: 500 条
- **语言**: 英语

## 数据格式

### queries.json 字段说明

| 字段 | 类型 | 说明 |
|------|------|------|
| `id` | string | 样本唯一标识符 |
| `audiocap_id` | string | AudioCaps 原始 ID |
| `youtube_id` | string | YouTube 视频 ID |

### 预测结果格式

```json
{
  "model_name": "your-model-name",
  "audio_to_text": {"audio_id": ["caption_id_1", "caption_id_2"]},
  "text_to_audio": {"caption_id": ["audio_id_1", "audio_id_2"]}
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
| **mAP** | 平均精度 |

## 数据来源

`queries.json` 基于 HuggingFace `AudioLLMs/audiocaps_test` 数据集格式生成 500 条样本。

原始数据集格式：
```python
{
    "id": "音频ID",
    "youtube_id": "YouTube视频ID",
    "start_time": 起始秒数,
    "caption": "音频描述文本"
}
```

## 参考资料

- [AudioCaps](https://audiocaps.github.io/)
