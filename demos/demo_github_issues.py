#!/usr/bin/env python3
"""
GitHub Issues - Issue 检索 Demo
================================

场景说明:
- 将 GitHub Issue 的标题和描述存入 Milvus
- 用户输入问题描述，检索相似的已有 Issue
- 适用于：重复 Issue 检测、相似问题查找、技术支持

数据集特点:
- 来自真实 GitHub 项目的 Issue 和 PR
- 包含标题、正文、评论、标签等信息
- 典型的技术问题描述风格
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
DATA_DIR = Path(__file__).parent.parent / "datasets" / "code" / "github_issues"
DB_PATH = str(Path(__file__).parent / "github_issues.db")
COLLECTION_NAME = "github_issues"
SAMPLE_SIZE = 1000


def load_data():
    """Load GitHub Issues data."""
    print("加载 GitHub Issues 数据...")
    df = pd.read_parquet(DATA_DIR / "train.parquet")

    # Filter issues (not PRs) with valid title and body
    df_filtered = df[
        (df["is_pull_request"] == False) &
        (df["title"].notna()) &
        (df["body"].notna()) &
        (df["body"].str.len() > 20)
    ].head(SAMPLE_SIZE)

    print(f"加载了 {len(df_filtered)} 个 Issue")
    return df_filtered


def extract_labels(labels):
    """Extract label names from labels list."""
    import numpy as np
    # Handle empty/None cases
    if labels is None:
        return "N/A"
    # Handle numpy arrays
    if isinstance(labels, np.ndarray):
        if labels.size == 0:
            return "N/A"
        labels = labels.tolist()
    # Handle empty lists
    if not labels:
        return "N/A"
    names = []
    for l in labels:
        if isinstance(l, dict) and l.get("name"):
            names.append(l["name"])
    return ", ".join(names[:3]) if names else "N/A"


def build_index(client, openai_client, issues_df):
    """Build Milvus index from issues."""
    print("\n正在构建索引...")

    # Combine title and body for embedding
    texts = [
        f"{row['title']}\n{row['body'][:500]}"
        for _, row in issues_df.iterrows()
    ]

    print("生成嵌入向量...")
    embeddings = get_embeddings(texts, openai_client)

    metadata = [
        {
            "title": row["title"][:200],
            "body": str(row["body"])[:1500] if row["body"] else "",
            "state": row["state"],
            "labels": extract_labels(row.get("labels")),
            "html_url": row.get("html_url", "N/A"),
        }
        for _, row in issues_df.iterrows()
    ]

    create_collection(client, COLLECTION_NAME, recreate=True)
    insert_data(client, COLLECTION_NAME, embeddings, metadata)

    print(f"已索引 {len(embeddings)} 个 Issue")


def search_issues(client, openai_client, query: str, top_k: int = 5):
    """Search for similar issues."""
    print(f"\n搜索问题: {query}")

    query_embedding = get_embeddings([query], openai_client)[0]

    results = search(
        client, COLLECTION_NAME, query_embedding,
        top_k=top_k,
        output_fields=["title", "body", "state", "labels", "html_url"]
    )

    # Custom print for issues
    print(f"\n{'='*60}")
    print(f"找到 {len(results)} 个相似 Issue:")
    print(f"{'='*60}")

    for i, result in enumerate(results, 1):
        similarity = result.get("distance", 0)
        entity = result.get("entity", result)
        title = entity.get("title", "N/A")
        body = entity.get("body", "N/A")
        state = entity.get("state", "N/A")
        labels = entity.get("labels", "N/A")

        if len(body) > 200:
            body = body[:200] + "..."

        print(f"\n[{i}] 相似度: {similarity:.4f} | 状态: {state}")
        print(f"    标签: {labels}")
        print(f"    标题: {title}")
        print(f"    描述: {body}")

    return results


def main():
    print("=" * 60)
    print("GitHub Issues 检索 Demo")
    print("=" * 60)

    openai_client = get_openai_client()
    milvus_client = get_milvus_client(DB_PATH)

    issues_df = load_data()
    build_index(milvus_client, openai_client, issues_df)

    # Demo searches
    demo_queries = [
        "TypeError when importing module",
        "Memory leak in long running process",
        "Documentation needs update for new API",
        "Feature request: add support for async",
    ]

    print("\n" + "=" * 60)
    print("示例搜索")
    print("=" * 60)

    for query in demo_queries:
        search_issues(milvus_client, openai_client, query, top_k=3)

    # Interactive mode
    print("\n" + "=" * 60)
    print("交互模式 (输入 'quit' 退出)")
    print("=" * 60)

    while True:
        query = input("\n请输入问题描述: ").strip()
        if query.lower() in ["quit", "exit", "q"]:
            break
        if query:
            search_issues(milvus_client, openai_client, query)

    print("\nDemo 结束!")


if __name__ == "__main__":
    main()
