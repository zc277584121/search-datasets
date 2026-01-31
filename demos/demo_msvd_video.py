#!/usr/bin/env python3
"""
MSVD - 视频描述检索 Demo
=========================

场景说明:
- 将视频描述文本存入 Milvus
- 用户输入描述，检索相似的视频描述
- 适用于：视频搜索、视频推荐、多模态检索的文本侧

数据集特点:
- 来自 YouTube 短视频的描述
- 每个视频有多条人工描述
- 约 40 条描述/视频
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
DATA_DIR = Path(__file__).parent.parent / "datasets" / "video" / "msvd"
DB_PATH = str(Path(__file__).parent / "msvd_video.db")
COLLECTION_NAME = "msvd_captions"
SAMPLE_SIZE = 1000


def load_data():
    """Load MSVD video caption data."""
    print("加载 MSVD 数据...")

    # Try different possible file names
    possible_files = ["train.parquet", "test.parquet", "validation.parquet"]
    df = None
    for fname in possible_files:
        fpath = DATA_DIR / fname
        if fpath.exists():
            df = pd.read_parquet(fpath)
            print(f"从 {fname} 加载数据")
            break

    if df is None:
        raise FileNotFoundError(f"No parquet files found in {DATA_DIR}")

    print(f"列名: {df.columns.tolist()}")
    df_sample = df.head(SAMPLE_SIZE)
    print(f"加载了 {len(df_sample)} 条视频描述")
    return df_sample


def build_index(client, openai_client, video_df):
    """Build Milvus index from video captions."""
    print("\n正在构建索引...")

    # Find the caption column
    caption_col = None
    for col in ["caption", "sentence", "text", "description"]:
        if col in video_df.columns:
            caption_col = col
            break

    if caption_col is None:
        print(f"Available columns: {video_df.columns.tolist()}")
        # Use first string column
        for col in video_df.columns:
            if video_df[col].dtype == "object":
                caption_col = col
                break

    texts = video_df[caption_col].astype(str).tolist()

    print("生成嵌入向量...")
    embeddings = get_embeddings(texts, openai_client)

    # Find video ID column
    video_id_col = None
    for col in ["video_id", "vid", "id", "video"]:
        if col in video_df.columns:
            video_id_col = col
            break

    metadata = []
    for i, (_, row) in enumerate(video_df.iterrows()):
        meta = {
            "caption": texts[i][:500],
        }
        if video_id_col:
            meta["video_id"] = str(row[video_id_col])
        else:
            meta["video_id"] = str(i)
        metadata.append(meta)

    create_collection(client, COLLECTION_NAME, recreate=True)
    insert_data(client, COLLECTION_NAME, embeddings, metadata)

    print(f"已索引 {len(embeddings)} 条描述")


def search_videos(client, openai_client, query: str, top_k: int = 5):
    """Search for similar video captions."""
    print(f"\n搜索描述: {query}")

    query_embedding = get_embeddings([query], openai_client)[0]

    results = search(
        client, COLLECTION_NAME, query_embedding,
        top_k=top_k,
        output_fields=["caption", "video_id"]
    )

    print_results(results, text_field="caption")
    return results


def main():
    print("=" * 60)
    print("MSVD 视频描述检索 Demo")
    print("=" * 60)

    openai_client = get_openai_client()
    milvus_client = get_milvus_client(DB_PATH)

    video_df = load_data()
    build_index(milvus_client, openai_client, video_df)

    # Demo searches
    demo_queries = [
        "A man is cooking in the kitchen",
        "Someone is playing guitar",
        "A cat is sleeping on the couch",
        "People are dancing at a party",
    ]

    print("\n" + "=" * 60)
    print("示例搜索")
    print("=" * 60)

    for query in demo_queries:
        search_videos(milvus_client, openai_client, query, top_k=3)

    # Interactive mode
    print("\n" + "=" * 60)
    print("交互模式 (输入 'quit' 退出)")
    print("=" * 60)

    while True:
        query = input("\n请输入视频描述: ").strip()
        if query.lower() in ["quit", "exit", "q"]:
            break
        if query:
            search_videos(milvus_client, openai_client, query)

    print("\nDemo 结束!")


if __name__ == "__main__":
    main()
