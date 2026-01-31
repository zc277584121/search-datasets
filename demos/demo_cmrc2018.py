#!/usr/bin/env python3
"""
CMRC 2018 - 中文阅读理解检索 Demo
==================================

场景说明:
- 将 CMRC 2018 的中文段落存入 Milvus
- 用户输入中文问题，检索最相关的段落
- 适用于：中文阅读理解、中文知识库检索

数据集特点:
- 中文 Wikipedia 段落
- 标准的中文抽取式问答数据集
- 包含 challenging 子集
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
DATA_DIR = Path(__file__).parent.parent / "datasets" / "chinese" / "cmrc2018"
DB_PATH = str(Path(__file__).parent / "cmrc2018.db")
COLLECTION_NAME = "cmrc2018_contexts"
SAMPLE_SIZE = 500


def load_data():
    """Load CMRC 2018 data."""
    print("加载 CMRC 2018 数据...")
    df = pd.read_parquet(DATA_DIR / "train.parquet")

    # Get unique contexts
    contexts_df = df[["context"]].drop_duplicates().head(SAMPLE_SIZE)
    print(f"加载了 {len(contexts_df)} 个唯一段落")

    return contexts_df


def build_index(client, openai_client, contexts_df):
    """Build Milvus index from contexts."""
    print("\n正在构建索引...")

    texts = contexts_df["context"].tolist()

    print("生成嵌入向量...")
    embeddings = get_embeddings(texts, openai_client)

    metadata = [
        {"context": row["context"][:2000]}
        for _, row in contexts_df.iterrows()
    ]

    create_collection(client, COLLECTION_NAME, recreate=True)
    insert_data(client, COLLECTION_NAME, embeddings, metadata)

    print(f"已索引 {len(embeddings)} 个段落")


def search_contexts(client, openai_client, query: str, top_k: int = 5):
    """Search for relevant contexts."""
    print(f"\n搜索问题: {query}")

    query_embedding = get_embeddings([query], openai_client)[0]

    results = search(
        client, COLLECTION_NAME, query_embedding,
        top_k=top_k, output_fields=["context"]
    )

    print_results(results, text_field="context")
    return results


def main():
    print("=" * 60)
    print("CMRC 2018 中文阅读理解检索 Demo")
    print("=" * 60)

    openai_client = get_openai_client()
    milvus_client = get_milvus_client(DB_PATH)

    contexts_df = load_data()
    build_index(milvus_client, openai_client, contexts_df)

    # Demo searches with Chinese queries
    demo_queries = [
        "北京有多少人口？",
        "长城是什么时候建造的？",
        "中国最大的城市是哪个？",
        "什么是人工智能？",
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
        query = input("\n请输入中文问题: ").strip()
        if query.lower() in ["quit", "exit", "q"]:
            break
        if query:
            search_contexts(milvus_client, openai_client, query)

    print("\nDemo 结束!")


if __name__ == "__main__":
    main()
