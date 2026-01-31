# WildChat Creative Writing 10K

## 数据集概述

WildChat Creative Writing 10K 是从 WildChat 数据集中筛选出的 10,000 条创意写作相关的 AI 对话记录。该数据集包含真实用户与大语言模型之间的创意写作交互，涵盖故事创作、角色扮演、诗歌写作等多种场景。

| 属性 | 值 |
|------|-----|
| **来源** | HuggingFace: `sam-paech/wildchat_creative_writing_annotated_10k` |
| **样本数** | 10,000 |
| **格式** | Parquet |
| **语言** | 多语言（以英语为主） |
| **任务类型** | 对话检索、创意写作分析、对话质量评估 |

## 数据集特点

- **真实对话**: 来自真实用户与 ChatGPT 等大语言模型的交互记录
- **创意写作聚焦**: 专门筛选创意写作相关对话
- **丰富标注**: 包含复杂度、创意性、质量评分等多维度标注
- **内容安全标注**: 包含 NSFW 和毒性标注

## 数据分割

| 分割 | 样本数 | 文件 |
|------|--------|------|
| train | 10,000 | `train.parquet` |

## 字段说明

| 字段名 | 类型 | 描述 |
|--------|------|------|
| `conversation_id` | string | 对话唯一标识符 |
| `model` | string | 使用的 AI 模型名称 |
| `timestamp` | string | 对话时间戳 |
| `conversation` | list | 对话内容列表，每条消息包含 content、language、redacted、role、toxic 字段 |
| `conversation_hash` | string | 对话内容哈希值 |
| `is_creative_writing` | bool | 是否为创意写作内容 |
| `complexity` | int | 复杂度评分 |
| `creativity` | int | 创意性评分 |
| `quality` | int | 质量评分 |
| `category_free` | string | 自由分类标签 |
| `category_constrained` | string | 约束分类标签 |
| `genre_free` | string | 自由体裁标签 |
| `genre_constrained` | string | 约束体裁标签 |
| `is_nsfw` | bool | 是否包含成人内容 |
| `nsfw_level` | int | NSFW 等级（0-5） |

### conversation 字段结构

每条对话消息包含以下字段：

| 字段 | 类型 | 描述 |
|------|------|------|
| `content` | string | 消息文本内容 |
| `language` | string | 消息语言 |
| `redacted` | bool | 是否经过脱敏处理 |
| `role` | string | 角色（user/assistant） |
| `toxic` | bool | 是否包含毒性内容 |

## 使用方法

```python
import pandas as pd

# 加载数据集
df = pd.read_parquet("train.parquet")

print(f"数据集大小: {len(df)}")
print(f"字段列表: {df.columns.tolist()}")

# 查看一条对话
sample = df.iloc[0]
print(f"对话ID: {sample['conversation_id']}")
print(f"模型: {sample['model']}")
print(f"创意性评分: {sample['creativity']}")
print(f"对话轮数: {len(sample['conversation'])}")

# 筛选高质量创意写作
high_quality = df[(df['quality'] >= 4) & (df['creativity'] >= 4)]
print(f"高质量创意对话数: {len(high_quality)}")
```

## 适用场景

1. **对话检索**: 基于对话内容或主题的检索任务
2. **创意写作分析**: 研究用户创意写作偏好和模式
3. **对话质量评估**: 训练对话质量评估模型
4. **内容安全研究**: 基于标注进行内容安全分析
5. **LLM 交互分析**: 研究用户与 LLM 的交互模式

## 注意事项

- 数据集可能包含敏感或不当内容，使用时请注意 `is_nsfw` 和 `toxic` 标注
- 对话内容可能经过部分脱敏处理（`redacted` 字段）
- 评分字段可用于数据筛选和质量控制

## 许可证

请参考原始数据集的许可证说明：[HuggingFace 页面](https://huggingface.co/datasets/sam-paech/wildchat_creative_writing_annotated_10k)

## 更新日期

2026-01-30
