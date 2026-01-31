#!/usr/bin/env python3
"""
COCO Karpathy - 多模态图文检索 Demo (CLIP)
============================================

场景说明:
- 使用 CLIP 模型同时编码图像和文本
- 支持：文本搜索图像、图像搜索相似图像
- 真正的多模态检索，不只是文本匹配

技术栈:
- Embedding: CLIP (openai/clip-vit-base-patch32)
- Vector DB: Milvus Lite
- 图像来源: COCO 图像 URL 在线下载
"""

import io
import sys
import requests
import pandas as pd
import torch
from pathlib import Path
from PIL import Image
from transformers import CLIPProcessor, CLIPModel
from pymilvus import MilvusClient

# Configuration
DATA_DIR = Path(__file__).parent.parent / "datasets" / "multimodal" / "coco_karpathy"
DB_PATH = str(Path(__file__).parent / "coco_multimodal.db")
COLLECTION_NAME = "coco_images"
SAMPLE_SIZE = 100  # Number of images to index (smaller for demo)
CLIP_MODEL = "openai/clip-vit-base-patch32"
CLIP_DIM = 512
DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"


class CLIPEmbedder:
    """CLIP model wrapper for image and text embedding."""

    def __init__(self, model_name: str = CLIP_MODEL, device: str = DEVICE):
        print(f"加载 CLIP 模型: {model_name} (device: {device})")
        self.device = device
        self.model = CLIPModel.from_pretrained(model_name).to(device)
        self.processor = CLIPProcessor.from_pretrained(model_name)
        self.model.eval()

    @torch.no_grad()
    def encode_images(self, images: list) -> list:
        """Encode a list of PIL images to embeddings."""
        inputs = self.processor(images=images, return_tensors="pt", padding=True)
        inputs = {k: v.to(self.device) for k, v in inputs.items() if isinstance(v, torch.Tensor)}
        outputs = self.model.get_image_features(**inputs)
        # Handle different output types
        if hasattr(outputs, 'pooler_output'):
            embeddings = outputs.pooler_output
        elif hasattr(outputs, 'last_hidden_state'):
            embeddings = outputs.last_hidden_state[:, 0]
        else:
            embeddings = outputs
        # Normalize embeddings
        embeddings = embeddings / embeddings.norm(dim=-1, keepdim=True)
        return embeddings.cpu().numpy().tolist()

    @torch.no_grad()
    def encode_texts(self, texts: list) -> list:
        """Encode a list of texts to embeddings."""
        inputs = self.processor(text=texts, return_tensors="pt", padding=True, truncation=True)
        inputs = {k: v.to(self.device) for k, v in inputs.items() if isinstance(v, torch.Tensor)}
        outputs = self.model.get_text_features(**inputs)
        # Handle different output types
        if hasattr(outputs, 'pooler_output'):
            embeddings = outputs.pooler_output
        elif hasattr(outputs, 'last_hidden_state'):
            embeddings = outputs.last_hidden_state[:, 0]
        else:
            embeddings = outputs
        # Normalize embeddings
        embeddings = embeddings / embeddings.norm(dim=-1, keepdim=True)
        return embeddings.cpu().numpy().tolist()


def download_image(url: str, timeout: int = 10) -> Image.Image:
    """Download image from URL."""
    try:
        response = requests.get(url, timeout=timeout)
        response.raise_for_status()
        return Image.open(io.BytesIO(response.content)).convert("RGB")
    except Exception as e:
        return None


def load_data():
    """Load COCO Karpathy data."""
    print("加载 COCO Karpathy 数据...")
    df = pd.read_parquet(DATA_DIR / "train.parquet")
    df_sample = df.head(SAMPLE_SIZE * 2)  # Load more in case some images fail
    print(f"加载了 {len(df_sample)} 条记录")
    return df_sample


def build_index(client: MilvusClient, embedder: CLIPEmbedder, df: pd.DataFrame):
    """Build Milvus index from images."""
    print("\n正在构建图像索引...")

    # Create collection
    if client.has_collection(COLLECTION_NAME):
        client.drop_collection(COLLECTION_NAME)

    client.create_collection(
        collection_name=COLLECTION_NAME,
        dimension=CLIP_DIM,
        metric_type="COSINE",
        auto_id=True,
    )

    # Process images in batches
    batch_size = 16
    indexed_count = 0
    target_count = SAMPLE_SIZE

    for i in range(0, len(df), batch_size):
        if indexed_count >= target_count:
            break

        batch_df = df.iloc[i:i + batch_size]
        images = []
        metadata = []

        for _, row in batch_df.iterrows():
            url = row.get("url")
            if not url:
                continue

            img = download_image(url)
            if img is None:
                continue

            images.append(img)
            # Get first caption
            captions = row.get("sentences", [])
            caption = str(captions[0]) if len(captions) > 0 else ""

            metadata.append({
                "image_url": url,
                "caption": caption[:500],
                "image_id": str(row.get("imgid", "")),
            })

        if not images:
            continue

        # Encode images
        embeddings = embedder.encode_images(images)

        # Insert to Milvus
        data = []
        for emb, meta in zip(embeddings, metadata):
            record = {"vector": emb}
            record.update(meta)
            data.append(record)

        client.insert(collection_name=COLLECTION_NAME, data=data)
        indexed_count += len(images)
        print(f"  已索引 {indexed_count}/{target_count} 张图像...")

    print(f"索引完成: {indexed_count} 张图像")


def search_by_text(client: MilvusClient, embedder: CLIPEmbedder, query: str, top_k: int = 5):
    """Search images by text query."""
    print(f"\n文本搜索: {query}")

    # Encode query text
    query_embedding = embedder.encode_texts([query])[0]

    # Search
    results = client.search(
        collection_name=COLLECTION_NAME,
        data=[query_embedding],
        limit=top_k,
        output_fields=["image_url", "caption", "image_id"]
    )

    print(f"\n{'='*60}")
    print(f"找到 {len(results[0])} 张相关图像:")
    print(f"{'='*60}")

    for i, result in enumerate(results[0], 1):
        similarity = result.get("distance", 0)
        entity = result.get("entity", result)
        url = entity.get("image_url", "N/A")
        caption = entity.get("caption", "N/A")

        print(f"\n[{i}] 相似度: {similarity:.4f}")
        print(f"    描述: {caption}")
        print(f"    URL: {url}")

    return results[0]


def search_by_image(client: MilvusClient, embedder: CLIPEmbedder, image_url: str, top_k: int = 5):
    """Search similar images by image URL."""
    print(f"\n图像搜索: {image_url}")

    # Download and encode query image
    img = download_image(image_url)
    if img is None:
        print("无法下载图像")
        return []

    query_embedding = embedder.encode_images([img])[0]

    # Search
    results = client.search(
        collection_name=COLLECTION_NAME,
        data=[query_embedding],
        limit=top_k + 1,  # +1 because query image might be in results
        output_fields=["image_url", "caption", "image_id"]
    )

    print(f"\n{'='*60}")
    print(f"找到 {len(results[0])} 张相似图像:")
    print(f"{'='*60}")

    for i, result in enumerate(results[0], 1):
        similarity = result.get("distance", 0)
        entity = result.get("entity", result)
        url = entity.get("image_url", "N/A")
        caption = entity.get("caption", "N/A")

        # Skip if same image
        if url == image_url:
            continue

        print(f"\n[{i}] 相似度: {similarity:.4f}")
        print(f"    描述: {caption}")
        print(f"    URL: {url}")

    return results[0]


def main():
    print("=" * 60)
    print("COCO 多模态图文检索 Demo (CLIP)")
    print("=" * 60)

    # Initialize
    embedder = CLIPEmbedder()
    milvus_client = MilvusClient(uri=DB_PATH)

    # Load and index data
    df = load_data()
    build_index(milvus_client, embedder, df)

    # Demo: Text to Image search
    text_queries = [
        "a dog playing with a ball",
        "people eating pizza at a restaurant",
        "a red car on the street",
        "a cat sleeping on a couch",
    ]

    print("\n" + "=" * 60)
    print("文本 → 图像 搜索示例")
    print("=" * 60)

    for query in text_queries:
        search_by_text(milvus_client, embedder, query, top_k=3)

    # Demo: Image to Image search (use first indexed image)
    print("\n" + "=" * 60)
    print("图像 → 图像 搜索示例")
    print("=" * 60)

    # Get a sample image URL from the collection
    sample_results = milvus_client.query(
        collection_name=COLLECTION_NAME,
        filter="",
        output_fields=["image_url"],
        limit=1
    )
    if sample_results:
        sample_url = sample_results[0].get("image_url")
        if sample_url:
            search_by_image(milvus_client, embedder, sample_url, top_k=3)

    # Interactive mode
    print("\n" + "=" * 60)
    print("交互模式")
    print("  输入文本进行图像搜索")
    print("  输入 'quit' 退出")
    print("=" * 60)

    while True:
        query = input("\n请输入搜索文本: ").strip()
        if query.lower() in ["quit", "exit", "q"]:
            break
        if query:
            search_by_text(milvus_client, embedder, query)

    print("\nDemo 结束!")


if __name__ == "__main__":
    main()
