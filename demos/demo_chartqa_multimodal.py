#!/usr/bin/env python3
"""
ChartQA - 多模态图表检索 Demo (CLIP)
======================================

场景说明:
- 使用 CLIP 模型编码图表图像和问题文本
- 支持：文本搜索相关图表、图表搜索相似图表
- 适用于：图表问答、数据可视化检索

技术栈:
- Embedding: CLIP (openai/clip-vit-base-patch32)
- Vector DB: Milvus Lite
- 图像格式: PNG/JPG 图表
"""

import io
import sys
import pandas as pd
import torch
from pathlib import Path
from PIL import Image
from transformers import CLIPProcessor, CLIPModel
from pymilvus import MilvusClient

# Configuration
DATA_DIR = Path(__file__).parent.parent / "datasets" / "multimodal" / "chartqa"
DB_PATH = str(Path(__file__).parent / "chartqa_multimodal.db")
COLLECTION_NAME = "chartqa_charts"
SAMPLE_SIZE = 200  # Number of charts to index
CLIP_MODEL = "openai/clip-vit-base-patch32"
CLIP_DIM = 512
DEVICE = "cuda:2" if torch.cuda.is_available() else "cpu"  # Use GPU 2


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
        if hasattr(outputs, 'pooler_output'):
            embeddings = outputs.pooler_output
        elif hasattr(outputs, 'last_hidden_state'):
            embeddings = outputs.last_hidden_state[:, 0]
        else:
            embeddings = outputs
        embeddings = embeddings / embeddings.norm(dim=-1, keepdim=True)
        return embeddings.cpu().numpy().tolist()

    @torch.no_grad()
    def encode_texts(self, texts: list) -> list:
        """Encode a list of texts to embeddings."""
        inputs = self.processor(text=texts, return_tensors="pt", padding=True, truncation=True)
        inputs = {k: v.to(self.device) for k, v in inputs.items() if isinstance(v, torch.Tensor)}
        outputs = self.model.get_text_features(**inputs)
        if hasattr(outputs, 'pooler_output'):
            embeddings = outputs.pooler_output
        elif hasattr(outputs, 'last_hidden_state'):
            embeddings = outputs.last_hidden_state[:, 0]
        else:
            embeddings = outputs
        embeddings = embeddings / embeddings.norm(dim=-1, keepdim=True)
        return embeddings.cpu().numpy().tolist()


def load_image_from_bytes(image_bytes: bytes) -> Image.Image:
    """Load image from bytes."""
    try:
        return Image.open(io.BytesIO(image_bytes)).convert("RGB")
    except Exception as e:
        return None


def load_data():
    """Load ChartQA data."""
    print("加载 ChartQA 数据...")
    df = pd.read_parquet(DATA_DIR / "train.parquet")
    df_sample = df.head(SAMPLE_SIZE)
    print(f"加载了 {len(df_sample)} 条图表记录")
    return df_sample


def build_index(client: MilvusClient, embedder: CLIPEmbedder, df: pd.DataFrame):
    """Build Milvus index from chart images."""
    print("\n正在构建图表索引...")

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

    for i in range(0, len(df), batch_size):
        batch_df = df.iloc[i:i + batch_size]
        images = []
        metadata = []

        for idx, (_, row) in enumerate(batch_df.iterrows()):
            image_data = row.get("image", {})
            if not isinstance(image_data, dict) or "bytes" not in image_data:
                continue

            img = load_image_from_bytes(image_data["bytes"])
            if img is None:
                continue

            images.append(img)

            # Get question and answer
            query = row.get("query", "")
            labels = row.get("label", [])
            label = str(labels[0]) if hasattr(labels, '__len__') and len(labels) > 0 else str(labels)
            source = "human" if row.get("human_or_machine", 0) == 0 else "machine"

            metadata.append({
                "question": str(query)[:300],
                "answer": label[:200],
                "source": source,
                "chart_id": str(i + idx),
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
        print(f"  已索引 {indexed_count}/{len(df)} 张图表...")

    print(f"索引完成: {indexed_count} 张图表")


def search_by_text(client: MilvusClient, embedder: CLIPEmbedder, query: str, top_k: int = 5):
    """Search charts by text query."""
    print(f"\n文本搜索: {query}")

    query_embedding = embedder.encode_texts([query])[0]

    results = client.search(
        collection_name=COLLECTION_NAME,
        data=[query_embedding],
        limit=top_k,
        output_fields=["question", "answer", "source", "chart_id"]
    )

    print(f"\n{'='*60}")
    print(f"找到 {len(results[0])} 张相关图表:")
    print(f"{'='*60}")

    for i, result in enumerate(results[0], 1):
        similarity = result.get("distance", 0)
        entity = result.get("entity", result)
        question = entity.get("question", "N/A")
        answer = entity.get("answer", "N/A")
        source = entity.get("source", "N/A")
        chart_id = entity.get("chart_id", "N/A")

        print(f"\n[{i}] 相似度: {similarity:.4f} | ID: {chart_id} | 来源: {source}")
        print(f"    问题: {question}")
        print(f"    答案: {answer}")

    return results[0]


def search_by_chart(client: MilvusClient, embedder: CLIPEmbedder, df: pd.DataFrame,
                    chart_idx: int, top_k: int = 5):
    """Search similar charts by chart index."""
    print(f"\n图表搜索: 索引 {chart_idx}")

    row = df.iloc[chart_idx]
    image_data = row.get("image", {})
    if not isinstance(image_data, dict) or "bytes" not in image_data:
        print("无法获取图表数据")
        return []

    img = load_image_from_bytes(image_data["bytes"])
    if img is None:
        print("无法加载图表")
        return []

    query_embedding = embedder.encode_images([img])[0]

    results = client.search(
        collection_name=COLLECTION_NAME,
        data=[query_embedding],
        limit=top_k + 1,
        output_fields=["question", "answer", "source", "chart_id"]
    )

    print(f"\n{'='*60}")
    print(f"原始图表问题: {row.get('query', 'N/A')}")
    print(f"找到 {len(results[0])} 张相似图表:")
    print(f"{'='*60}")

    count = 0
    for result in results[0]:
        similarity = result.get("distance", 0)
        entity = result.get("entity", result)
        result_chart_id = entity.get("chart_id", "N/A")

        if result_chart_id == str(chart_idx):
            continue

        count += 1
        question = entity.get("question", "N/A")
        answer = entity.get("answer", "N/A")

        print(f"\n[{count}] 相似度: {similarity:.4f} | ID: {result_chart_id}")
        print(f"    问题: {question}")
        print(f"    答案: {answer}")

    return results[0]


def main():
    print("=" * 60)
    print("ChartQA 多模态图表检索 Demo (CLIP)")
    print("=" * 60)

    embedder = CLIPEmbedder()
    milvus_client = MilvusClient(uri=DB_PATH)

    df = load_data()
    build_index(milvus_client, embedder, df)

    # Demo: Text to Chart search
    text_queries = [
        "bar chart showing sales data",
        "pie chart with percentage distribution",
        "line graph showing trends over time",
        "chart about revenue and profit",
        "graph comparing different categories",
    ]

    print("\n" + "=" * 60)
    print("文本 → 图表 搜索示例")
    print("=" * 60)

    for query in text_queries:
        search_by_text(milvus_client, embedder, query, top_k=3)

    # Demo: Chart to Chart search
    print("\n" + "=" * 60)
    print("图表 → 图表 搜索示例")
    print("=" * 60)

    search_by_chart(milvus_client, embedder, df, chart_idx=0, top_k=3)

    # Interactive mode
    print("\n" + "=" * 60)
    print("交互模式")
    print("  输入图表描述进行搜索")
    print("  输入 'quit' 退出")
    print("=" * 60)

    while True:
        query = input("\n请输入图表描述: ").strip()
        if query.lower() in ["quit", "exit", "q"]:
            break
        if query:
            search_by_text(milvus_client, embedder, query)

    print("\nDemo 结束!")


if __name__ == "__main__":
    main()
