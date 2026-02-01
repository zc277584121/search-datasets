# Discord 聊天检索提交格式

## 任务说明

从 Discord 聊天记录中检索与查询相关的消息。

## 输入

评估数据 `datasets/conversation/discord_chat/`：
- `data`: 聊天消息文本

## 输出格式

提交文件 `predictions.json`，格式如下：

```json
{
  "model_name": "你的模型名称",
  "model_description": "模型简要描述",
  "queries": [
    "有人一起打游戏吗",
    "今天服务器怎么了",
    "推荐一些好歌"
  ],
  "predictions": [
    {
      "query": "有人一起打游戏吗",
      "retrieved": [
        "anyone down for some ranked?",
        "looking for teammates for val",
        "who's playing tonight",
        "need one more for the squad",
        "let's queue up"
      ]
    },
    {
      "query": "今天服务器怎么了",
      "retrieved": [
        "server is lagging so bad rn",
        "is discord down for anyone else",
        "can't connect to voice",
        "ping is through the roof",
        "having connection issues"
      ]
    }
  ]
}
```

### 字段说明

| 字段 | 类型 | 说明 |
|------|------|------|
| `queries` | list | 查询列表 |
| `predictions` | list | 每个查询的检索结果 |
| `predictions[].query` | string | 查询文本 |
| `predictions[].retrieved` | list | 检索到的消息列表（按相关性排序） |

### 注意事项

- 每个查询返回 5-10 条最相关的消息
- 消息按相关性**降序**排列

## 评估指标

使用 **LLM-as-Judge** 评估：
- **Mean Relevance**: 平均相关性评分（1-5分）

## 运行评估

```bash
# 需要设置 OPENAI_API_KEY 环境变量
python eval/run_eval.py --task discord --submission submissions/discord/predictions.json
```
