#!/usr/bin/env python3
"""
Discord Chat - 聊天消息检索 Demo
=================================

场景说明:
- 将 Discord 聊天消息存入 Milvus
- 用户输入关键词/话题，检索相关聊天消息
- 适用于：聊天记录搜索、社交媒体分析、对话检索

数据集特点:
- 来自 Discord 平台的真实聊天记录
- 非正式的即时通讯风格
- 包含俚语、缩写等口语化表达
"""

import sys
import pandas as pd
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from utils import (
    get_openai_client, get_embeddings, get_milvus_client,
    create_collection, insert_data, search, print_results
)

# Configuration
DATA_DIR = Path(__file__).parent.parent / "datasets" / "conversation" / "discord_chat"
DB_PATH = str(Path(__file__).parent / "discord_chat.db")
COLLECTION_NAME = "discord_messages"
SAMPLE_SIZE = 2000


def load_data():
    """Load Discord chat data."""
    print("加载 Discord Chat 数据...")
    df = pd.read_parquet(DATA_DIR / "train.parquet")

    # Filter messages with sufficient length
    df_filtered = df[
        df["data"].notna() &
        (df["data"].str.len() > 20) &
        (df["data"].str.len() < 2000)  # Avoid very long messages
    ].head(SAMPLE_SIZE)

    print(f"加载了 {len(df_filtered)} 条聊天消息")
    return df_filtered


def build_index(client, openai_client, chat_df):
    """Build Milvus index from chat messages."""
    print("\n正在构建索引...")

    texts = chat_df["data"].tolist()

    print("生成嵌入向量...")
    embeddings = get_embeddings(texts, openai_client)

    metadata = [
        {"message": row["data"][:1000]}
        for _, row in chat_df.iterrows()
    ]

    create_collection(client, COLLECTION_NAME, recreate=True)
    insert_data(client, COLLECTION_NAME, embeddings, metadata)

    print(f"已索引 {len(embeddings)} 条消息")


def search_messages(client, openai_client, query: str, top_k: int = 5):
    """Search for similar chat messages."""
    print(f"\n搜索话题: {query}")

    query_embedding = get_embeddings([query], openai_client)[0]

    results = search(
        client, COLLECTION_NAME, query_embedding,
        top_k=top_k,
        output_fields=["message"]
    )

    print_results(results, text_field="message")
    return results


def main():
    print("=" * 60)
    print("Discord Chat 聊天消息检索 Demo")
    print("=" * 60)

    openai_client = get_openai_client()
    milvus_client = get_milvus_client(DB_PATH)

    chat_df = load_data()
    build_index(milvus_client, openai_client, chat_df)

    # Demo searches
    demo_queries = [
        "looking for someone to play games",
        "how do I fix this bug",
        "what do you think about",
        "anyone want to join voice chat",
    ]

    print("\n" + "=" * 60)
    print("示例搜索")
    print("=" * 60)

    for query in demo_queries:
        search_messages(milvus_client, openai_client, query, top_k=3)

    # Interactive mode
    print("\n" + "=" * 60)
    print("交互模式 (输入 'quit' 退出)")
    print("=" * 60)

    while True:
        query = input("\n请输入搜索话题: ").strip()
        if query.lower() in ["quit", "exit", "q"]:
            break
        if query:
            search_messages(milvus_client, openai_client, query)

    print("\nDemo 结束!")


if __name__ == "__main__":
    main()
