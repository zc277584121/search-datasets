# 任务：MSVD 视频文本检索

## 任务描述

构建一个视频-文本检索系统，能够：
1. **视频 → 文本**：给定一个视频，找到最相关的文本描述
2. **文本 → 视频**：给定一条文本描述，找到最相关的视频

## 数据集信息

| 属性 | 值 |
|------|-----|
| **名称** | MSVD (Microsoft Video Description) |
| **来源** | `friedrichor/MSVD` |
| **语言** | 英语 |
| **许可证** | 研究用途 |
| **规模** | ~2K 视频，~80K 描述（每视频约 40 条） |

### 数据字段

| 字段 | 类型 | 描述 |
|------|------|------|
| `video_id` | string | YouTube 视频 ID 带时间戳 |
| `video` | bytes/path | 视频数据或路径 |
| `caption` | string | 视频描述 |
| `start_time` | float | 片段起始时间 |
| `end_time` | float | 片段结束时间 |

### 数据集特点

- **短视频**：平均约 10 秒
- **多描述**：每个视频约 40 条来自不同标注者的描述
- **多样内容**：动物、烹饪、运动、互动、音乐

## 评估目标

达到：
- **视频→文本 R@1 ≥ 40%**
- **文本→视频 R@1 ≥ 30%**
- 双向 **R@10 ≥ 80%**

## 评估方法

### 检索指标

```python
import numpy as np

def compute_video_text_retrieval_metrics(
    video_embeddings,
    text_embeddings,
    video_ids,
    text_video_ids,
    k_values=[1, 5, 10]
):
    """
    计算视频-文本检索指标

    参数:
        video_embeddings: (N_videos, D) 视频特征矩阵
        text_embeddings: (N_texts, D) 文本特征矩阵
        video_ids: 唯一视频 ID
        text_video_ids: 每条文本对应的视频 ID（用于匹配）
    """
    # 计算相似度
    similarity = np.dot(text_embeddings, video_embeddings.T)  # (N_texts, N_videos)

    # 构建标准答案映射
    video_to_idx = {vid: i for i, vid in enumerate(video_ids)}

    results = {'text_to_video': {}, 'video_to_text': {}}

    # 文本 → 视频
    for k in k_values:
        correct = 0
        for i, text_vid in enumerate(text_video_ids):
            top_k = np.argsort(-similarity[i])[:k]
            gt_idx = video_to_idx[text_vid]
            if gt_idx in top_k:
                correct += 1
        results['text_to_video'][f'R@{k}'] = 100.0 * correct / len(text_video_ids)

    # 视频 → 文本（对于每个视频，其任何描述都是正确的）
    for k in k_values:
        correct = 0
        for vid_idx, vid in enumerate(video_ids):
            top_k = np.argsort(-similarity.T[vid_idx])[:k]
            gt_indices = [i for i, tvid in enumerate(text_video_ids) if tvid == vid]
            if any(idx in top_k for idx in gt_indices):
                correct += 1
        results['video_to_text'][f'R@{k}'] = 100.0 * correct / len(video_ids)

    return results

def compute_median_rank(similarity_matrix, ground_truth_indices):
    """计算标准答案的中位排名"""
    ranks = []
    for i, gt in enumerate(ground_truth_indices):
        sorted_indices = np.argsort(-similarity_matrix[i])
        for rank, idx in enumerate(sorted_indices):
            if idx in gt:
                ranks.append(rank + 1)
                break
    return np.median(ranks)
```

## 建议方案

1. **基于 CLIP**：使用 CLIP 配合帧采样

2. **视频语言模型**：VideoCLIP、InternVideo、LanguageBind

3. **帧聚合**：独立编码帧，使用池化/注意力聚合

4. **时序建模**：使用 3D 卷积或视频 Transformer

## 视频处理

```python
import cv2
import torch
import numpy as np

def extract_frames(video_path, num_frames=8):
    """从视频中均匀提取帧"""
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    indices = np.linspace(0, total_frames - 1, num_frames, dtype=int)
    frames = []

    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if ret:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame)

    cap.release()
    return frames

def sample_frames_uniform(video_tensor, num_frames=8):
    """从视频张量中均匀采样帧"""
    T = video_tensor.shape[0]
    indices = torch.linspace(0, T - 1, num_frames).long()
    return video_tensor[indices]
```

## 核心挑战

- 跨帧的时间理解
- 处理可变视频长度
- 视频编码的计算成本
- 每个视频有多个有效描述

## 基线性能

| 模型 | V2T R@1 | T2V R@1 |
|------|---------|---------|
| CLIP (均值池化) | 35.2 | 24.8 |
| CLIP4Clip | 43.1 | 32.5 |
| InternVideo | 55.2 | 42.8 |

## 参考资料

- 论文: [Collecting Highly Parallel Data for Paraphrase Evaluation](https://aclanthology.org/P11-1020/)
- HuggingFace: https://huggingface.co/datasets/friedrichor/MSVD
