#!/usr/bin/env python3
"""
ELI5 Reddit - 问答社区检索 Demo
================================

场景说明:
- 将 Reddit ELI5 的问题和最佳答案存入 Milvus
- 用户输入问题，检索相似问题及其答案
- 适用于：FAQ 检索、知识问答、社区问答搜索

数据集特点:
- 来自 Reddit r/explainlikeimfive
- 答案以简单易懂的方式解释复杂概念
- 包含答案投票分数，可筛选高质量答案
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
DATA_DIR = Path(__file__).parent.parent / "datasets" / "conversation" / "eli5_reddit"
DB_PATH = str(Path(__file__).parent / "eli5_reddit.db")
COLLECTION_NAME = "eli5_questions"
SAMPLE_SIZE = 1000  # Number of Q&A pairs to index


def load_data():
    """Load ELI5 Reddit data with best answers."""
    print("加载 ELI5 Reddit 数据...")
    df = pd.read_parquet(DATA_DIR / "train.parquet")

    # Process data: get question + best answer
    records = []
    for _, row in df.head(SAMPLE_SIZE * 2).iterrows():  # Load more to filter
        answers = row["answers"]
        if answers and "text" in answers and "score" in answers:
            # Get the best answer (highest score)
            scores = answers["score"]
            texts = answers["text"]
            # Handle numpy arrays
            if hasattr(scores, '__len__') and len(scores) > 0 and hasattr(texts, '__len__') and len(texts) > 0:
                import numpy as np
                scores_list = list(scores) if isinstance(scores, np.ndarray) else scores
                texts_list = list(texts) if isinstance(texts, np.ndarray) else texts
                best_idx = scores_list.index(max(scores_list))
                best_answer = texts_list[best_idx]
                best_score = scores_list[best_idx]

                # Only include high-quality answers
                if best_score >= 10 and len(best_answer) > 50:
                    records.append({
                        "question": row["title"],
                        "answer": best_answer,
                        "score": best_score,
                        "subreddit": row["subreddit"]
                    })

                    if len(records) >= SAMPLE_SIZE:
                        break

    result_df = pd.DataFrame(records)
    print(f"加载了 {len(result_df)} 个高质量问答对")
    return result_df


def build_index(client, openai_client, qa_df):
    """Build Milvus index from Q&A pairs."""
    print("\n正在构建索引...")

    # Embed questions for question-to-question matching
    texts = qa_df["question"].tolist()

    print("生成嵌入向量...")
    embeddings = get_embeddings(texts, openai_client)

    metadata = [
        {
            "question": row["question"][:500],
            "answer": row["answer"][:2000],
            "score": int(row["score"]),
        }
        for _, row in qa_df.iterrows()
    ]

    create_collection(client, COLLECTION_NAME, recreate=True)
    insert_data(client, COLLECTION_NAME, embeddings, metadata)

    print(f"已索引 {len(embeddings)} 个问答对")


def search_qa(client, openai_client, query: str, top_k: int = 5):
    """Search for similar questions and their answers."""
    print(f"\n搜索问题: {query}")

    query_embedding = get_embeddings([query], openai_client)[0]

    results = search(
        client, COLLECTION_NAME, query_embedding,
        top_k=top_k, output_fields=["question", "answer", "score"]
    )

    # Custom print for Q&A format
    print(f"\n{'='*60}")
    print(f"找到 {len(results)} 个相似问题:")
    print(f"{'='*60}")

    for i, result in enumerate(results, 1):
        similarity = result.get("distance", 0)
        entity = result.get("entity", result)
        question = entity.get("question", "N/A")
        answer = entity.get("answer", "N/A")
        score = entity.get("score", 0)

        if len(answer) > 300:
            answer = answer[:300] + "..."

        print(f"\n[{i}] 相似度: {similarity:.4f} | 投票分数: {score}")
        print(f"    问题: {question}")
        print(f"    答案: {answer}")

    return results


def main():
    print("=" * 60)
    print("ELI5 Reddit 问答社区检索 Demo")
    print("=" * 60)

    openai_client = get_openai_client()
    milvus_client = get_milvus_client(DB_PATH)

    qa_df = load_data()
    build_index(milvus_client, openai_client, qa_df)

    # Demo searches
    demo_queries = [
        "Why is the sky blue?",
        "How do airplanes fly?",
        "What causes earthquakes?",
        "Why do we dream?",
    ]

    print("\n" + "=" * 60)
    print("示例搜索")
    print("=" * 60)

    for query in demo_queries:
        search_qa(milvus_client, openai_client, query, top_k=3)

    # Interactive mode
    print("\n" + "=" * 60)
    print("交互模式 (输入 'quit' 退出)")
    print("=" * 60)

    while True:
        query = input("\n请输入问题: ").strip()
        if query.lower() in ["quit", "exit", "q"]:
            break
        if query:
            search_qa(milvus_client, openai_client, query)

    print("\nDemo 结束!")


if __name__ == "__main__":
    main()
