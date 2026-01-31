#!/usr/bin/env python3
"""
Enron Email - 企业邮件检索 Demo
================================

场景说明:
- 将 Enron 邮件的主题和正文存入 Milvus
- 用户输入关键词/描述，检索相关邮件
- 适用于：企业邮件搜索、文档检索、电子取证

数据集特点:
- 来自真实企业的邮件通讯
- 包含垃圾邮件/正常邮件标注
- 经典的文本检索基准数据集
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
DATA_DIR = Path(__file__).parent.parent / "datasets" / "document" / "enron_mini"
DB_PATH = str(Path(__file__).parent / "enron_email.db")
COLLECTION_NAME = "enron_emails"
SAMPLE_SIZE = 1000


def load_data():
    """Load Enron email data."""
    print("加载 Enron 邮件数据...")
    df = pd.read_parquet(DATA_DIR / "train.parquet")

    # Filter ham emails (non-spam) with valid content
    df_filtered = df[
        (df["label"] == 0) &  # ham = 0, spam = 1
        (df["subject"].notna()) &
        (df["message"].notna()) &
        (df["message"].str.len() > 50)
    ].head(SAMPLE_SIZE)

    print(f"加载了 {len(df_filtered)} 封正常邮件")
    return df_filtered


def build_index(client, openai_client, emails_df):
    """Build Milvus index from emails."""
    print("\n正在构建索引...")

    # Combine subject and message for embedding
    texts = [
        f"Subject: {row['subject']}\n{row['message'][:800]}"
        for _, row in emails_df.iterrows()
    ]

    print("生成嵌入向量...")
    embeddings = get_embeddings(texts, openai_client)

    metadata = [
        {
            "subject": str(row["subject"])[:200] if row["subject"] else "",
            "message": str(row["message"])[:2000] if row["message"] else "",
            "date": str(row["date"]) if pd.notna(row.get("date")) else "Unknown",
        }
        for _, row in emails_df.iterrows()
    ]

    create_collection(client, COLLECTION_NAME, recreate=True)
    insert_data(client, COLLECTION_NAME, embeddings, metadata)

    print(f"已索引 {len(embeddings)} 封邮件")


def search_emails(client, openai_client, query: str, top_k: int = 5):
    """Search for relevant emails."""
    print(f"\n搜索查询: {query}")

    query_embedding = get_embeddings([query], openai_client)[0]

    results = search(
        client, COLLECTION_NAME, query_embedding,
        top_k=top_k,
        output_fields=["subject", "message", "date"]
    )

    # Custom print for emails
    print(f"\n{'='*60}")
    print(f"找到 {len(results)} 封相关邮件:")
    print(f"{'='*60}")

    for i, result in enumerate(results, 1):
        similarity = result.get("distance", 0)
        entity = result.get("entity", result)
        subject = entity.get("subject", "N/A")
        message = entity.get("message", "N/A")
        date = entity.get("date", "N/A")

        if len(message) > 300:
            message = message[:300] + "..."

        print(f"\n[{i}] 相似度: {similarity:.4f}")
        print(f"    日期: {date}")
        print(f"    主题: {subject}")
        print(f"    内容: {message}")

    return results


def main():
    print("=" * 60)
    print("Enron Email 企业邮件检索 Demo")
    print("=" * 60)

    openai_client = get_openai_client()
    milvus_client = get_milvus_client(DB_PATH)

    emails_df = load_data()
    build_index(milvus_client, openai_client, emails_df)

    # Demo searches
    demo_queries = [
        "meeting schedule tomorrow",
        "project deadline extension",
        "budget approval request",
        "quarterly report summary",
    ]

    print("\n" + "=" * 60)
    print("示例搜索")
    print("=" * 60)

    for query in demo_queries:
        search_emails(milvus_client, openai_client, query, top_k=3)

    # Interactive mode
    print("\n" + "=" * 60)
    print("交互模式 (输入 'quit' 退出)")
    print("=" * 60)

    while True:
        query = input("\n请输入搜索关键词: ").strip()
        if query.lower() in ["quit", "exit", "q"]:
            break
        if query:
            search_emails(milvus_client, openai_client, query)

    print("\nDemo 结束!")


if __name__ == "__main__":
    main()
