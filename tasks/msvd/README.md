# MSVD 视频文本检索

## 任务描述

MSVD 评估模型在视频-文本双向检索任务上的能力。

## 数据集信息

- **来源**: `friedrichor/MSVD`
- **评测集**: 500 条
- **语言**: 英语

## 数据格式

### queries.json 字段说明

| 字段 | 类型 | 说明 |
|------|------|------|
| `id` | string | 样本唯一标识符 |
| `video_id` | string | 视频 ID |

### 预测结果格式

```json
{
  "model_name": "your-model-name",
  "video_to_text": {"video_id": ["caption_id_1", "caption_id_2"]},
  "text_to_video": {"caption_id": ["video_id_1", "video_id_2"]}
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
| **Median Rank** | 中位排名 |

## 参考资料

- [MSVD Dataset](https://www.cs.utexas.edu/users/ml/clamp/videoDescription/)
