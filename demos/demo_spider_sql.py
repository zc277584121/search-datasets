#!/usr/bin/env python3
"""
Spider - NL2SQL 问题检索 Demo
==============================

场景说明:
- 将自然语言问题和对应的 SQL 查询存入 Milvus
- 用户输入问题，检索相似问题及其 SQL 实现
- 适用于：SQL 辅助生成、查询推荐、NL2SQL 参考

数据集特点:
- 200个数据库，138个领域
- 自然语言问题和对应 SQL 查询
- 跨域零样本评测基准
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
DATA_DIR = Path(__file__).parent.parent / "datasets" / "table" / "spider"
DB_PATH = str(Path(__file__).parent / "spider_sql.db")
COLLECTION_NAME = "spider_queries"
SAMPLE_SIZE = 1000


def load_data():
    """Load Spider NL2SQL data."""
    print("加载 Spider 数据...")
    df = pd.read_parquet(DATA_DIR / "train.parquet")
    print(f"列名: {df.columns.tolist()}")

    df_sample = df.head(SAMPLE_SIZE)
    print(f"加载了 {len(df_sample)} 个问题-SQL 对")
    return df_sample


def build_index(client, openai_client, spider_df):
    """Build Milvus index from NL questions."""
    print("\n正在构建索引...")

    # Find question column
    question_col = None
    for col in ["question", "query", "nl", "natural_language"]:
        if col in spider_df.columns:
            question_col = col
            break

    if question_col is None:
        question_col = spider_df.columns[0]

    texts = spider_df[question_col].astype(str).tolist()

    print("生成嵌入向量...")
    embeddings = get_embeddings(texts, openai_client)

    # Find SQL column
    sql_col = None
    for col in ["query", "sql", "SQL"]:
        if col in spider_df.columns and col != question_col:
            sql_col = col
            break

    # Find database column
    db_col = None
    for col in ["db_id", "database", "db"]:
        if col in spider_df.columns:
            db_col = col
            break

    metadata = []
    for i, (_, row) in enumerate(spider_df.iterrows()):
        meta = {
            "question": texts[i][:500],
        }
        if sql_col:
            meta["sql"] = str(row[sql_col])[:500]
        if db_col:
            meta["database"] = str(row[db_col])
        metadata.append(meta)

    create_collection(client, COLLECTION_NAME, recreate=True)
    insert_data(client, COLLECTION_NAME, embeddings, metadata)

    print(f"已索引 {len(embeddings)} 个问题")


def search_queries(client, openai_client, query: str, top_k: int = 5):
    """Search for similar NL questions and their SQL."""
    print(f"\n搜索问题: {query}")

    query_embedding = get_embeddings([query], openai_client)[0]

    results = search(
        client, COLLECTION_NAME, query_embedding,
        top_k=top_k,
        output_fields=["question", "sql", "database"]
    )

    # Custom print for SQL results
    print(f"\n{'='*60}")
    print(f"找到 {len(results)} 个相似问题:")
    print(f"{'='*60}")

    for i, result in enumerate(results, 1):
        similarity = result.get("distance", 0)
        entity = result.get("entity", result)
        question = entity.get("question", "N/A")
        sql = entity.get("sql", "N/A")
        database = entity.get("database", "N/A")

        print(f"\n[{i}] 相似度: {similarity:.4f} | 数据库: {database}")
        print(f"    问题: {question}")
        print(f"    SQL: {sql}")

    return results


def main():
    print("=" * 60)
    print("Spider NL2SQL 问题检索 Demo")
    print("=" * 60)

    openai_client = get_openai_client()
    milvus_client = get_milvus_client(DB_PATH)

    spider_df = load_data()
    build_index(milvus_client, openai_client, spider_df)

    # Demo searches
    demo_queries = [
        "How many students are there?",
        "Find all employees with salary greater than 50000",
        "What is the average price of products?",
        "List all orders from last month",
    ]

    print("\n" + "=" * 60)
    print("示例搜索")
    print("=" * 60)

    for query in demo_queries:
        search_queries(milvus_client, openai_client, query, top_k=3)

    # Interactive mode
    print("\n" + "=" * 60)
    print("交互模式 (输入 'quit' 退出)")
    print("=" * 60)

    while True:
        query = input("\n请输入数据库查询问题: ").strip()
        if query.lower() in ["quit", "exit", "q"]:
            break
        if query:
            search_queries(milvus_client, openai_client, query)

    print("\nDemo 结束!")


if __name__ == "__main__":
    main()
