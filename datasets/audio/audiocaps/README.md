# AudioCaps 数据集

## 概述

**AudioCaps** 是一个大规模音频描述数据集，包含人工标注的音频-文本对，用于音频描述生成和音频-文本检索任务。

- **发布机构**: Google Research
- **论文**: "AudioCaps: Generating Captions for Audios in The Wild" (NAACL 2019)
- **HuggingFace**: `AudioLLMs/audiocaps_test`（测试集）/ `d0rj/audiocaps`（完整）
- **许可证**: 研究用途
- **语言**: 英语

## 数据集特点

- **基于 AudioSet**: 音频来自 YouTube 的 AudioSet 数据集
- **众包标注**: 通过 Amazon Mechanical Turk 收集描述
- **10 秒片段**: 每个音频片段固定 10 秒
- **野外音频**: 真实世界的多样音频场景

## 数据集规模

| 子集 | 音频数 | 描述数 | 说明 |
|------|--------|--------|------|
| train | 49,838 | 49,838 | 训练集（每音频 1 描述） |
| validation | 2,480 | 2,480 | 验证集 |
| test | 4,875 | 4,875 | 测试集 |

**总计**: ~57,000 个音频-文本对

**注**: 本目录使用的是 `AudioLLMs/audiocaps_test`，仅包含测试集，体积约 1.3GB。

## 数据字段说明

| 字段名 | 类型 | 说明 |
|--------|------|------|
| `audio` | Audio | 音频数据（采样率 16kHz） |
| `audiocap_id` | int | AudioCaps ID |
| `youtube_id` | string | YouTube 视频 ID |
| `start_time` | int | 片段起始时间（秒） |
| `caption` | string | 音频描述文本 |

## 数据示例

```python
{
    "audiocap_id": 2,
    "youtube_id": "---lTs1dxhU",
    "start_time": 27,
    "caption": "A woman speaks and then a man answers",
    "audio": {
        "path": "audio.wav",
        "array": [...],  # 音频波形数据
        "sampling_rate": 16000
    }
}
```

## 音频内容类别

AudioCaps 涵盖 AudioSet 的多种声音类别：

| 大类 | 示例 |
|------|------|
| 人声 | 说话、唱歌、笑声、咳嗽 |
| 音乐 | 吉他、钢琴、鼓、管弦乐 |
| 自然 | 雨声、风声、鸟鸣、雷声 |
| 动物 | 狗叫、猫叫、马嘶 |
| 机械 | 汽车、飞机、机器运转 |
| 环境 | 人群、交通、室内 |

## 评测任务

### 1. 音频描述生成 (Audio Captioning)
- **输入**: 10 秒音频片段
- **输出**: 描述性文本
- **指标**: BLEU, METEOR, CIDEr, SPICE, SPIDEr

### 2. 音频-文本检索 (Audio-Text Retrieval)
- **输入**: 音频或文本查询
- **输出**: 最相关的文本或音频
- **指标**: R@1, R@5, R@10, mAP

## 评测指标说明

| 指标 | 说明 |
|------|------|
| BLEU | n-gram 精确匹配 |
| METEOR | 考虑同义词和词干 |
| CIDEr | TF-IDF 加权的共识度量 |
| SPICE | 基于场景图的语义相似度 |
| SPIDEr | SPICE + CIDEr 的平均 |

## 使用方法

```python
from datasets import load_dataset
import soundfile as sf

# 加载测试集
dataset = load_dataset("AudioLLMs/audiocaps_test")

# 查看示例
example = dataset["test"][0]
print("Caption:", example["caption"])

# 获取音频数据
audio = example["audio"]
print("Sampling rate:", audio["sampling_rate"])
print("Duration:", len(audio["array"]) / audio["sampling_rate"], "seconds")

# 保存音频文件
sf.write("sample.wav", audio["array"], audio["sampling_rate"])

# 播放音频（在 notebook 中）
from IPython.display import Audio
Audio(audio["array"], rate=audio["sampling_rate"])
```

## 音频下载说明

完整数据集需要从 YouTube 下载音频：

```bash
# 安装依赖
pip install yt-dlp ffmpeg

# 下载脚本示例
yt-dlp -x --audio-format wav \
    --postprocessor-args "-ss START_TIME -t 10" \
    -o "audio/%(id)s.%(ext)s" \
    "https://youtube.com/watch?v=YOUTUBE_ID"
```

**注意**: 部分 YouTube 视频可能已被删除，实际可下载数量可能少于标注数量。

## 与其他音频数据集比较

| 数据集 | 音频数 | 描述/音频 | 时长 | 特点 |
|--------|--------|-----------|------|------|
| **AudioCaps** | 57K | 1 | 10s | 大规模，野外 |
| Clotho | 7K | 5 | 15-30s | 高质量描述 |
| WavCaps | 400K | 1 | 变长 | 最大规模 |
| MACS | 3.9K | 2-5 | 10s | 多标注 |

## 应用场景

1. **音频理解模型训练**: 学习音频语义理解
2. **语音助手**: 描述周围环境声音
3. **辅助技术**: 帮助听障人士理解声音
4. **音频检索**: 用文本搜索音频
5. **多模态学习**: 音频-文本对齐

## 参考链接

- 官网: https://audiocaps.github.io/
- 论文: https://aclanthology.org/N19-1011/
- GitHub: https://github.com/cdjkim/audiocaps
- HuggingFace: https://huggingface.co/datasets/AudioLLMs/audiocaps_test
