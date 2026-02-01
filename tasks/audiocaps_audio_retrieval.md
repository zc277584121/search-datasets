# 任务：AudioCaps 音频文本检索

## 任务描述

构建一个**音频和文本**的跨模态检索系统：
1. **音频 → 文本**：给定一段音频片段，找到最相关的文本描述
2. **文本 → 音频**：给定一条文本描述，找到最相关的音频片段

## 数据集信息

| 属性 | 值 |
|------|-----|
| **名称** | AudioCaps |
| **来源** | `AudioLLMs/audiocaps_test` |
| **语言** | 英语 |
| **许可证** | 研究用途 |
| **规模** | 测试集 ~4.9K 样本（10 秒片段） |

### 数据字段

| 字段 | 类型 | 描述 |
|------|------|------|
| `audio` | Audio | 音频波形（16kHz 采样率） |
| `audiocap_id` | int | AudioCaps ID |
| `youtube_id` | string | 来源 YouTube 视频 ID |
| `start_time` | int | 片段起始时间（秒） |
| `caption` | string | 人工编写的音频描述 |

### 音频类别

涵盖多种声音类型：
- **人声**：说话、唱歌、笑声、咳嗽
- **音乐**：各种乐器、流派
- **自然**：雨声、风声、鸟鸣、雷声
- **动物**：狗叫、猫叫、马嘶
- **机械**：汽车、机器、引擎
- **环境**：人群、交通、室内声音

## 评估目标

达到：
- **音频→文本 R@1 ≥ 30%**
- **文本→音频 R@1 ≥ 25%**
- 双向 **R@10 ≥ 70%**

## 评估方法

### 检索指标

```python
import numpy as np

def compute_recall_at_k(similarity_matrix, k_values=[1, 5, 10]):
    """
    计算检索 Recall@K

    参数:
        similarity_matrix: (N, N) 成对相似度，对角线是正确答案
        k_values: 要计算的 K 值列表
    """
    N = similarity_matrix.shape[0]
    results = {}

    for k in k_values:
        correct = 0
        for i in range(N):
            # 获取 top-k 索引
            top_k = np.argsort(-similarity_matrix[i])[:k]
            if i in top_k:
                correct += 1
        results[f'R@{k}'] = 100.0 * correct / N

    return results

def evaluate_audio_text_retrieval(audio_embeddings, text_embeddings):
    """
    评估双向音频-文本检索
    假设一一对应（audio[i] 匹配 text[i]）
    """
    # 归一化向量
    audio_embeddings = audio_embeddings / np.linalg.norm(audio_embeddings, axis=1, keepdims=True)
    text_embeddings = text_embeddings / np.linalg.norm(text_embeddings, axis=1, keepdims=True)

    # 计算相似度矩阵
    similarity = np.dot(audio_embeddings, text_embeddings.T)

    # 音频 → 文本
    a2t_metrics = compute_recall_at_k(similarity)

    # 文本 → 音频
    t2a_metrics = compute_recall_at_k(similarity.T)

    return {
        'audio_to_text': a2t_metrics,
        'text_to_audio': t2a_metrics
    }
```

### 音频描述生成指标（替代方案）

如果生成描述而非检索：

```python
from pycocoevalcap.cider.cider import Cider
from pycocoevalcap.spice.spice import Spice

def evaluate_captioning(predictions, references):
    """
    评估音频描述生成质量

    参数:
        predictions: dict {audio_id: [predicted_caption]}
        references: dict {audio_id: [reference_captions]}
    """
    # CIDEr 分数
    cider_scorer = Cider()
    cider_score, _ = cider_scorer.compute_score(references, predictions)

    # SPICE 分数
    spice_scorer = Spice()
    spice_score, _ = spice_scorer.compute_score(references, predictions)

    return {
        'CIDEr': cider_score,
        'SPICE': spice_score,
        'SPIDEr': (cider_score + spice_score) / 2
    }
```

## 建议方案

1. **CLAP**：对比语言-音频预训练（LAION-CLAP、Microsoft CLAP）

2. **AudioLLM**：Qwen-Audio、SALMONN 或类似的音频语言模型

3. **音频编码器 + 文本编码器**：组合音频编码器（AST、BEATs）和文本编码器

4. **基于频谱图**：将音频转换为频谱图，使用视觉语言模型

## 核心挑战

- 多样化的音频类型（语音、音乐、环境声）
- 时间理解（事件序列）
- 细粒度声音区分
- 处理复杂场景中的重叠声音

## 音频处理

```python
import librosa
import torch

def load_audio(audio_path, sr=16000, duration=10):
    """加载并预处理音频"""
    waveform, _ = librosa.load(audio_path, sr=sr, duration=duration)

    # 填充或截断到固定长度
    target_length = sr * duration
    if len(waveform) < target_length:
        waveform = np.pad(waveform, (0, target_length - len(waveform)))
    else:
        waveform = waveform[:target_length]

    return torch.tensor(waveform)
```

## 参考资料

- 论文: [AudioCaps: Generating Captions for Audios in The Wild](https://aclanthology.org/N19-1011/)
- CLAP: [CLAP: Learning Audio Concepts from Natural Language Supervision](https://arxiv.org/abs/2206.04769)
