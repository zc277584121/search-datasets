#!/usr/bin/env python3
"""
SQuAD 2.0 - 文本问答检索 Demo
==================================

场景说明:
- 将 SQuAD 2.0 的上下文段落(context)存入 Milvus
- 用户输入问题，检索最相关的段落
- 适用于：阅读理解、文档问答、知识库检索

数据集特点:
- 包含 Wikipedia 段落和相关问题
- 每个段落可能对应多个问题
- 部分问题是不可回答的（答案不在段落中）
"""

import sys
import pandas as pd
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from utils import (
    get_openai_client, get_embeddings, get_milvus_client,
    create_collection, insert_data, search, print_results
)

# Configuration
DATA_DIR = Path(__file__).parent.parent / "datasets" / "text" / "squad2"
DB_PATH = str(Path(__file__).parent / "squad2.db")
COLLECTION_NAME = "squad2_contexts"
SAMPLE_SIZE = 500  # Number of unique contexts to index (for demo)


def load_data():
    """Load SQuAD 2.0 data and extract unique contexts."""
    print("加载 SQuAD 2.0 数据...")
    df = pd.read_parquet(DATA_DIR / "train.parquet")

    # Get unique contexts (each context appears multiple times with different questions)
    contexts_df = df[["title", "context"]].drop_duplicates().head(SAMPLE_SIZE)
    print(f"加载了 {len(contexts_df)} 个唯一段落")

    return contexts_df


def build_index(client, openai_client, contexts_df):
    """Build Milvus index from contexts."""
    print("\n正在构建索引...")

    # Prepare texts for embedding
    texts = contexts_df["context"].tolist()

    # Generate embeddings
    print("生成嵌入向量...")
    embeddings = get_embeddings(texts, openai_client)

    # Prepare metadata
    metadata = [
        {
            "title": row["title"],
            "context": row["context"][:2000]  # Truncate for storage
        }
        for _, row in contexts_df.iterrows()
    ]

    # Create collection and insert data
    create_collection(client, COLLECTION_NAME, recreate=True)
    insert_data(client, COLLECTION_NAME, embeddings, metadata)

    print(f"已索引 {len(embeddings)} 个段落")


def search_contexts(client, openai_client, query: str, top_k: int = 5):
    """Search for relevant contexts given a question."""
    print(f"\n搜索问题: {query}")

    # Generate query embedding
    query_embedding = get_embeddings([query], openai_client)[0]

    # Search
    results = search(
        client, COLLECTION_NAME, query_embedding,
        top_k=top_k, output_fields=["title", "context"]
    )

    print_results(results, text_field="context")
    return results


def main():
    print("=" * 60)
    print("SQuAD 2.0 文本问答检索 Demo")
    print("=" * 60)

    # Initialize clients
    openai_client = get_openai_client()
    milvus_client = get_milvus_client(DB_PATH)

    # Load and index data
    contexts_df = load_data()
    build_index(milvus_client, openai_client, contexts_df)

    # Demo searches
    demo_queries = [
        "When did Beyonce start becoming popular?",
        "What is the capital of France?",
        "How does photosynthesis work?",
        "Who won the Super Bowl in 2015?",
    ]

    print("\n" + "=" * 60)
    print("示例搜索")
    print("=" * 60)

    for query in demo_queries:
        search_contexts(milvus_client, openai_client, query, top_k=3)

    # Interactive mode
    print("\n" + "=" * 60)
    print("交互模式 (输入 'quit' 退出)")
    print("=" * 60)

    while True:
        query = input("\n请输入问题: ").strip()
        if query.lower() in ["quit", "exit", "q"]:
            break
        if query:
            search_contexts(milvus_client, openai_client, query)

    print("\nDemo 结束!")


if __name__ == "__main__":
    main()
