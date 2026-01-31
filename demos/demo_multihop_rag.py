#!/usr/bin/env python3
"""
MultiHop-RAG - 多跳检索增强生成 Demo
=====================================

场景说明:
- 将新闻文档语料库存入 Milvus
- 用户输入复杂问题，检索相关文档
- 适用于：RAG 系统、多文档问答、新闻检索

数据集特点:
- 来自多个新闻来源的英文文章
- 问题通常需要结合多个文档才能回答
- 适合构建 RAG 系统的检索组件
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
DATA_DIR = Path(__file__).parent.parent / "datasets" / "rag" / "multihop_rag"
DB_PATH = str(Path(__file__).parent / "multihop_rag.db")
COLLECTION_NAME = "news_corpus"
SAMPLE_SIZE = 500


def load_data():
    """Load MultiHop-RAG corpus."""
    print("加载 MultiHop-RAG 语料库...")
    df = pd.read_parquet(DATA_DIR / "corpus_train.parquet")
    df = df.head(SAMPLE_SIZE)
    print(f"加载了 {len(df)} 篇文档")
    return df


def build_index(client, openai_client, corpus_df):
    """Build Milvus index from news corpus."""
    print("\n正在构建索引...")

    # Use title + body for embedding
    texts = [
        f"{row['title']}\n{row['body'][:1000]}"
        for _, row in corpus_df.iterrows()
    ]

    print("生成嵌入向量...")
    embeddings = get_embeddings(texts, openai_client)

    metadata = [
        {
            "title": row["title"],
            "body": row["body"][:2000],
            "source": row.get("source", "Unknown"),
            "category": row.get("category", "Unknown"),
        }
        for _, row in corpus_df.iterrows()
    ]

    create_collection(client, COLLECTION_NAME, recreate=True)
    insert_data(client, COLLECTION_NAME, embeddings, metadata)

    print(f"已索引 {len(embeddings)} 篇文档")


def search_documents(client, openai_client, query: str, top_k: int = 5):
    """Search for relevant documents."""
    print(f"\n搜索查询: {query}")

    query_embedding = get_embeddings([query], openai_client)[0]

    results = search(
        client, COLLECTION_NAME, query_embedding,
        top_k=top_k, output_fields=["title", "body", "source", "category"]
    )

    print_results(results, text_field="title")
    return results


def main():
    print("=" * 60)
    print("MultiHop-RAG 多跳检索增强生成 Demo")
    print("=" * 60)

    openai_client = get_openai_client()
    milvus_client = get_milvus_client(DB_PATH)

    corpus_df = load_data()
    build_index(milvus_client, openai_client, corpus_df)

    # Demo searches
    demo_queries = [
        "What are the best Black Friday deals on Amazon?",
        "Latest technology news about AI",
        "Climate change and environmental policies",
        "Stock market and economic trends",
    ]

    print("\n" + "=" * 60)
    print("示例搜索")
    print("=" * 60)

    for query in demo_queries:
        search_documents(milvus_client, openai_client, query, top_k=3)

    # Interactive mode
    print("\n" + "=" * 60)
    print("交互模式 (输入 'quit' 退出)")
    print("=" * 60)

    while True:
        query = input("\n请输入查询: ").strip()
        if query.lower() in ["quit", "exit", "q"]:
            break
        if query:
            search_documents(milvus_client, openai_client, query)

    print("\nDemo 结束!")


if __name__ == "__main__":
    main()
