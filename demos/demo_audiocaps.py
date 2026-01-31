#!/usr/bin/env python3
"""
AudioCaps - 音频描述检索 Demo
==============================

场景说明:
- 将音频描述文本存入 Milvus
- 用户输入声音描述，检索相似的音频描述
- 适用于：音频搜索、声音效果库检索、多模态检索的文本侧

数据集特点:
- 来自 AudioSet (YouTube) 的音频描述
- 10秒音频片段的人工描述
- 丰富的环境音、音乐、语音描述
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
DATA_DIR = Path(__file__).parent.parent / "datasets" / "audio" / "audiocaps"
DB_PATH = str(Path(__file__).parent / "audiocaps.db")
COLLECTION_NAME = "audiocaps"
SAMPLE_SIZE = 1000


def load_data():
    """Load AudioCaps data."""
    print("加载 AudioCaps 数据...")

    # Try different possible file names
    possible_files = ["test.parquet", "train.parquet", "validation.parquet"]
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
    print(f"加载了 {len(df_sample)} 条音频描述")
    return df_sample


def build_index(client, openai_client, audio_df):
    """Build Milvus index from audio captions."""
    print("\n正在构建索引...")

    # Find the caption column - this dataset uses 'answer' for the caption
    caption_col = None
    for col in ["answer", "caption", "text", "description", "sentence"]:
        if col in audio_df.columns:
            # Skip 'context' as it contains raw audio bytes
            if col != "context":
                caption_col = col
                break

    if caption_col is None:
        # Use first string column that's not 'context'
        for col in audio_df.columns:
            if col != "context" and audio_df[col].dtype == "object":
                caption_col = col
                break

    print(f"Using column '{caption_col}' for captions")
    texts = audio_df[caption_col].astype(str).tolist()

    print("生成嵌入向量...")
    embeddings = get_embeddings(texts, openai_client)

    # For this dataset, we'll just use row index as ID
    metadata = []
    for i, (_, row) in enumerate(audio_df.iterrows()):
        meta = {
            "caption": texts[i][:500],
            "audio_id": str(i),
        }
        # Add instruction if available
        if "instruction" in row and row["instruction"]:
            meta["instruction"] = str(row["instruction"])[:200]
        metadata.append(meta)

    create_collection(client, COLLECTION_NAME, recreate=True)
    insert_data(client, COLLECTION_NAME, embeddings, metadata)

    print(f"已索引 {len(embeddings)} 条描述")


def search_audio(client, openai_client, query: str, top_k: int = 5):
    """Search for similar audio captions."""
    print(f"\n搜索描述: {query}")

    query_embedding = get_embeddings([query], openai_client)[0]

    results = search(
        client, COLLECTION_NAME, query_embedding,
        top_k=top_k,
        output_fields=["caption", "audio_id"]
    )

    print_results(results, text_field="caption")
    return results


def main():
    print("=" * 60)
    print("AudioCaps 音频描述检索 Demo")
    print("=" * 60)

    openai_client = get_openai_client()
    milvus_client = get_milvus_client(DB_PATH)

    audio_df = load_data()
    build_index(milvus_client, openai_client, audio_df)

    # Demo searches
    demo_queries = [
        "A dog barking loudly",
        "Rain falling on a roof",
        "Someone playing piano",
        "Cars honking in traffic",
        "Birds singing in the morning",
    ]

    print("\n" + "=" * 60)
    print("示例搜索")
    print("=" * 60)

    for query in demo_queries:
        search_audio(milvus_client, openai_client, query, top_k=3)

    # Interactive mode
    print("\n" + "=" * 60)
    print("交互模式 (输入 'quit' 退出)")
    print("=" * 60)

    while True:
        query = input("\n请输入声音描述: ").strip()
        if query.lower() in ["quit", "exit", "q"]:
            break
        if query:
            search_audio(milvus_client, openai_client, query)

    print("\nDemo 结束!")


if __name__ == "__main__":
    main()
