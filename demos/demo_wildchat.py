#!/usr/bin/env python3
"""
WildChat 10K - AI 对话检索 Demo
================================

场景说明:
- 将 AI 创意写作对话存入 Milvus
- 用户输入主题/描述，检索相关的 AI 对话
- 适用于：对话检索、创意写作参考、AI 交互分析

数据集特点:
- 真实用户与 ChatGPT 等 LLM 的对话
- 聚焦创意写作场景
- 包含质量、创意性等多维度评分
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
DATA_DIR = Path(__file__).parent.parent / "datasets" / "conversation" / "wildchat_10k"
DB_PATH = str(Path(__file__).parent / "wildchat.db")
COLLECTION_NAME = "wildchat_conversations"
SAMPLE_SIZE = 500


def load_data():
    """Load WildChat conversation data."""
    print("加载 WildChat 10K 数据...")
    df = pd.read_parquet(DATA_DIR / "train.parquet")

    # Filter high-quality creative writing conversations
    df_filtered = df[
        (df["is_creative_writing"] == True) &
        (df["quality"] >= 3) &
        (df["is_nsfw"] == False)
    ].head(SAMPLE_SIZE)

    print(f"加载了 {len(df_filtered)} 个高质量创意写作对话")
    return df_filtered


def extract_conversation_text(conversation):
    """Extract text from conversation list."""
    if conversation is None:
        return ""
    # Handle numpy arrays
    if hasattr(conversation, '__len__') and len(conversation) == 0:
        return ""
    texts = []
    for msg in list(conversation)[:3]:  # Take first 3 turns
        if isinstance(msg, dict):
            role = msg.get("role", "")
            content = msg.get("content", "")
        else:
            role = "unknown"
            content = str(msg)
        if content:
            texts.append(f"[{role}]: {content[:200]}")
    return "\n".join(texts)


def build_index(client, openai_client, conv_df):
    """Build Milvus index from conversations."""
    print("\n正在构建索引...")

    # Extract first user message for embedding (the prompt)
    texts = []
    for _, row in conv_df.iterrows():
        conv = row["conversation"]
        # Handle numpy arrays and lists
        if conv is not None and hasattr(conv, '__len__') and len(conv) > 0:
            first_msg = conv[0]
            if isinstance(first_msg, dict):
                first_msg = first_msg.get("content", "")[:500]
            else:
                first_msg = str(first_msg)[:500]
            texts.append(first_msg)
        else:
            texts.append("")

    print("生成嵌入向量...")
    embeddings = get_embeddings(texts, openai_client)

    metadata = []
    for _, row in conv_df.iterrows():
        conv_text = extract_conversation_text(row["conversation"])
        metadata.append({
            "conversation_preview": conv_text[:1500],
            "category": row.get("category_free", "Unknown"),
            "genre": row.get("genre_free", "Unknown"),
            "quality": int(row.get("quality", 0)),
            "creativity": int(row.get("creativity", 0)),
        })

    create_collection(client, COLLECTION_NAME, recreate=True)
    insert_data(client, COLLECTION_NAME, embeddings, metadata)

    print(f"已索引 {len(embeddings)} 个对话")


def search_conversations(client, openai_client, query: str, top_k: int = 5):
    """Search for similar conversations."""
    print(f"\n搜索主题: {query}")

    query_embedding = get_embeddings([query], openai_client)[0]

    results = search(
        client, COLLECTION_NAME, query_embedding,
        top_k=top_k,
        output_fields=["conversation_preview", "category", "genre", "quality", "creativity"]
    )

    # Custom print
    print(f"\n{'='*60}")
    print(f"找到 {len(results)} 个相关对话:")
    print(f"{'='*60}")

    for i, result in enumerate(results, 1):
        similarity = result.get("distance", 0)
        entity = result.get("entity", result)
        preview = entity.get("conversation_preview", "N/A")
        category = entity.get("category", "N/A")
        genre = entity.get("genre", "N/A")
        quality = entity.get("quality", 0)
        creativity = entity.get("creativity", 0)

        if len(preview) > 400:
            preview = preview[:400] + "..."

        print(f"\n[{i}] 相似度: {similarity:.4f}")
        print(f"    类别: {category} | 体裁: {genre}")
        print(f"    质量: {quality}/5 | 创意: {creativity}/5")
        print(f"    对话预览:\n    {preview}")

    return results


def main():
    print("=" * 60)
    print("WildChat 10K AI 对话检索 Demo")
    print("=" * 60)

    openai_client = get_openai_client()
    milvus_client = get_milvus_client(DB_PATH)

    conv_df = load_data()
    build_index(milvus_client, openai_client, conv_df)

    # Demo searches
    demo_queries = [
        "Write a fantasy story about a dragon",
        "Help me write a romantic poem",
        "Create a science fiction plot about time travel",
        "Write a mystery story with a detective",
    ]

    print("\n" + "=" * 60)
    print("示例搜索")
    print("=" * 60)

    for query in demo_queries:
        search_conversations(milvus_client, openai_client, query, top_k=3)

    # Interactive mode
    print("\n" + "=" * 60)
    print("交互模式 (输入 'quit' 退出)")
    print("=" * 60)

    while True:
        query = input("\n请输入创意写作主题: ").strip()
        if query.lower() in ["quit", "exit", "q"]:
            break
        if query:
            search_conversations(milvus_client, openai_client, query)

    print("\nDemo 结束!")


if __name__ == "__main__":
    main()
