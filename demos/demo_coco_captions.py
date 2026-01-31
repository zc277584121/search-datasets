#!/usr/bin/env python3
"""
COCO Karpathy - 图文检索 Demo (文本部分)
=========================================

场景说明:
- 将 COCO 图像描述(caption)存入 Milvus
- 用户输入图像描述，检索相似描述对应的图像信息
- 适用于：图文检索、图像搜索、多模态检索的文本侧

数据集特点:
- 来自 MS COCO 数据集的图像描述
- 每张图像有多条人工描述
- 标准的图文检索基准

注意：此 demo 仅处理文本描述，图像检索需要额外的视觉模型
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
DATA_DIR = Path(__file__).parent.parent / "datasets" / "multimodal" / "coco_karpathy"
DB_PATH = str(Path(__file__).parent / "coco_captions.db")
COLLECTION_NAME = "coco_captions"
SAMPLE_SIZE = 2000


def load_data():
    """Load COCO Karpathy caption data."""
    print("加载 COCO Karpathy 数据...")
    df = pd.read_parquet(DATA_DIR / "train.parquet")

    # Sample captions
    df_sample = df.head(SAMPLE_SIZE)
    print(f"加载了 {len(df_sample)} 条图像描述")
    return df_sample


def build_index(client, openai_client, captions_df):
    """Build Milvus index from captions."""
    print("\n正在构建索引...")

    # Get captions - handle different possible column names
    if "sentences" in captions_df.columns:
        # Extract first caption from sentences array (numpy array of strings)
        texts = []
        for _, row in captions_df.iterrows():
            sentences = row.get("sentences")
            if sentences is not None and len(sentences) > 0:
                # sentences is a numpy array of strings
                texts.append(str(sentences[0]))
            else:
                texts.append("")
    elif "caption" in captions_df.columns:
        texts = captions_df["caption"].tolist()
    else:
        # Try to find any text column
        text_cols = [c for c in captions_df.columns if "caption" in c.lower() or "text" in c.lower()]
        if text_cols:
            texts = captions_df[text_cols[0]].tolist()
        else:
            print(f"Available columns: {captions_df.columns.tolist()}")
            raise ValueError("Cannot find caption column")

    print("生成嵌入向量...")
    embeddings = get_embeddings(texts, openai_client)

    metadata = []
    for i, (_, row) in enumerate(captions_df.iterrows()):
        meta = {
            "caption": texts[i][:500] if texts[i] else "",
            "image_id": str(row.get("imgid", row.get("image_id", i))),
        }
        # Add image path if available
        if "filepath" in row:
            meta["image_path"] = str(row["filepath"])
        elif "filename" in row:
            meta["image_path"] = str(row["filename"])
        metadata.append(meta)

    create_collection(client, COLLECTION_NAME, recreate=True)
    insert_data(client, COLLECTION_NAME, embeddings, metadata)

    print(f"已索引 {len(embeddings)} 条描述")


def search_captions(client, openai_client, query: str, top_k: int = 5):
    """Search for similar captions."""
    print(f"\n搜索描述: {query}")

    query_embedding = get_embeddings([query], openai_client)[0]

    results = search(
        client, COLLECTION_NAME, query_embedding,
        top_k=top_k,
        output_fields=["caption", "image_id", "image_path"]
    )

    # Custom print
    print(f"\n{'='*60}")
    print(f"找到 {len(results)} 条相似描述:")
    print(f"{'='*60}")

    for i, result in enumerate(results, 1):
        similarity = result.get("distance", 0)
        entity = result.get("entity", result)
        caption = entity.get("caption", "N/A")
        image_id = entity.get("image_id", "N/A")
        image_path = entity.get("image_path", "N/A")

        print(f"\n[{i}] 相似度: {similarity:.4f}")
        print(f"    图像ID: {image_id}")
        print(f"    描述: {caption}")
        if image_path and image_path != "N/A":
            print(f"    路径: {image_path}")

    return results


def main():
    print("=" * 60)
    print("COCO Karpathy 图文检索 Demo (文本侧)")
    print("=" * 60)

    openai_client = get_openai_client()
    milvus_client = get_milvus_client(DB_PATH)

    captions_df = load_data()
    build_index(milvus_client, openai_client, captions_df)

    # Demo searches
    demo_queries = [
        "A dog playing in the park",
        "People eating food at a restaurant",
        "A car parked on the street",
        "Children playing with toys",
    ]

    print("\n" + "=" * 60)
    print("示例搜索")
    print("=" * 60)

    for query in demo_queries:
        search_captions(milvus_client, openai_client, query, top_k=3)

    # Interactive mode
    print("\n" + "=" * 60)
    print("交互模式 (输入 'quit' 退出)")
    print("=" * 60)

    while True:
        query = input("\n请输入图像描述: ").strip()
        if query.lower() in ["quit", "exit", "q"]:
            break
        if query:
            search_captions(milvus_client, openai_client, query)

    print("\nDemo 结束!")


if __name__ == "__main__":
    main()
