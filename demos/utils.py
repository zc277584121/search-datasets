"""
Shared utilities for Milvus search demos.
共享工具模块，用于 Milvus 搜索 demo。

This module provides:
- OpenAI embedding generation (text-embedding-3-small)
- Milvus Lite client wrapper using MilvusClient interface
"""

import os
from typing import List, Optional
from openai import OpenAI
from pymilvus import MilvusClient, DataType

# OpenAI embedding model
EMBEDDING_MODEL = "text-embedding-3-small"
EMBEDDING_DIM = 1536  # text-embedding-3-small dimension


def get_openai_client() -> OpenAI:
    """
    Get OpenAI client. Expects OPENAI_API_KEY in environment.
    获取 OpenAI 客户端，需要环境变量 OPENAI_API_KEY。
    """
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError(
            "OPENAI_API_KEY environment variable is required. "
            "请设置环境变量 OPENAI_API_KEY"
        )
    return OpenAI(api_key=api_key)


def get_embeddings(texts: List[str], client: Optional[OpenAI] = None) -> List[List[float]]:
    """
    Generate embeddings for a list of texts using OpenAI text-embedding-3-small.
    使用 OpenAI text-embedding-3-small 模型生成文本嵌入向量。

    Args:
        texts: List of texts to embed (文本列表)
        client: Optional OpenAI client (可选的 OpenAI 客户端)

    Returns:
        List of embedding vectors (嵌入向量列表)
    """
    if client is None:
        client = get_openai_client()

    # OpenAI API has a limit of 8191 tokens per request, batch if needed
    # Process in batches to avoid rate limits
    batch_size = 100
    all_embeddings = []

    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        # Filter out empty strings
        batch = [t if t else " " for t in batch]

        response = client.embeddings.create(
            model=EMBEDDING_MODEL,
            input=batch
        )
        batch_embeddings = [item.embedding for item in response.data]
        all_embeddings.extend(batch_embeddings)

    return all_embeddings


def get_milvus_client(db_path: str) -> MilvusClient:
    """
    Get Milvus Lite client using MilvusClient interface.
    获取 Milvus Lite 客户端（使用 MilvusClient 接口）。

    Args:
        db_path: Path to the .db file (Milvus Lite 数据库文件路径，以 .db 结尾)

    Returns:
        MilvusClient instance (MilvusClient 实例)
    """
    return MilvusClient(uri=db_path)


def create_collection(
    client: MilvusClient,
    collection_name: str,
    dim: int = EMBEDDING_DIM,
    recreate: bool = False
) -> None:
    """
    Create a collection with vector field.
    创建带有向量字段的 collection。

    Args:
        client: MilvusClient instance
        collection_name: Name of the collection
        dim: Vector dimension (default: 1536 for text-embedding-3-small)
        recreate: Whether to drop and recreate if exists
    """
    if client.has_collection(collection_name):
        if recreate:
            client.drop_collection(collection_name)
        else:
            return

    # Create collection with auto-id
    client.create_collection(
        collection_name=collection_name,
        dimension=dim,
        metric_type="COSINE",
        auto_id=True,
    )


def insert_data(
    client: MilvusClient,
    collection_name: str,
    vectors: List[List[float]],
    metadata: List[dict]
) -> None:
    """
    Insert vectors with metadata into collection.
    向 collection 插入向量和元数据。

    Args:
        client: MilvusClient instance
        collection_name: Name of the collection
        vectors: List of embedding vectors
        metadata: List of metadata dicts (one per vector)
    """
    # Combine vectors and metadata
    data = []
    for vec, meta in zip(vectors, metadata):
        record = {"vector": vec}
        record.update(meta)
        data.append(record)

    client.insert(collection_name=collection_name, data=data)


def search(
    client: MilvusClient,
    collection_name: str,
    query_vector: List[float],
    top_k: int = 5,
    output_fields: Optional[List[str]] = None
) -> List[dict]:
    """
    Search for similar vectors.
    搜索相似向量。

    Args:
        client: MilvusClient instance
        collection_name: Name of the collection
        query_vector: Query embedding vector
        top_k: Number of results to return
        output_fields: Fields to return (None = all)

    Returns:
        List of search results with metadata
    """
    results = client.search(
        collection_name=collection_name,
        data=[query_vector],
        limit=top_k,
        output_fields=output_fields
    )
    return results[0] if results else []


def print_results(results: List[dict], text_field: str = "text") -> None:
    """
    Pretty print search results.
    格式化打印搜索结果。

    Args:
        results: Search results from Milvus
        text_field: Name of the text field to display
    """
    print(f"\n{'='*60}")
    print(f"找到 {len(results)} 条结果:")
    print(f"{'='*60}")

    for i, result in enumerate(results, 1):
        score = result.get("distance", 0)
        entity = result.get("entity", result)
        text = entity.get(text_field, "N/A")

        # Truncate long text
        if isinstance(text, str) and len(text) > 200:
            text = text[:200] + "..."

        print(f"\n[{i}] 相似度: {score:.4f}")
        print(f"    {text_field}: {text}")

        # Print other fields
        for key, value in entity.items():
            if key not in [text_field, "vector"]:
                if isinstance(value, str) and len(value) > 100:
                    value = value[:100] + "..."
                print(f"    {key}: {value}")
