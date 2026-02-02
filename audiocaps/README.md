# AudioCaps 音频描述检索

## 任务描述

AudioCaps 音频描述检索任务评估模型检索相似音频描述的能力。给定一段音频描述（caption），从语料库中检索出语义相似的描述。

## 数据集信息

- **来源**: `d0rj/audiocaps` (HuggingFace)
- **评测集**: 500 条 queries（从 validation split 采样）
- **语料库**: 49,838 条 captions（train split）
- **语言**: 英语

## 数据格式

### queries.json 字段说明

| 字段 | 类型 | 说明 |
|------|------|------|
| `id` | string | 查询唯一标识符 |
| `youtube_id` | string | YouTube 视频 ID |
| `start_time` | int | 音频开始时间（秒） |
| `caption` | string | 音频描述文本 |

### corpus.json 字段说明

| 字段 | 类型 | 说明 |
|------|------|------|
| `id` | string | 文档唯一标识符 |
| `youtube_id` | string | YouTube 视频 ID |
| `start_time` | int | 音频开始时间（秒） |
| `caption` | string | 音频描述文本 |

### 预测结果格式

```json
{
  "model_name": "your-model-name",
  "predictions": {
    "0": ["corpus_id_1", "corpus_id_2", "corpus_id_3"],
    "1": ["corpus_id_5", "corpus_id_8"]
  }
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
| **MRR** | 平均倒数排名 |
| **MAP** | 平均精度均值 |
| **Recall@K** | Top-K 召回率 |

## 数据来源

`queries.json` 从 HuggingFace `d0rj/audiocaps` 数据集的 validation split 采样生成。`corpus.json` 使用 train split 的全部数据。Ground truth 通过关键词匹配生成（匹配度 >= 2 个关键词）。

## 参考资料

- [AudioCaps Paper](https://arxiv.org/abs/1903.00048)
- [HuggingFace Dataset](https://huggingface.co/datasets/d0rj/audiocaps)
