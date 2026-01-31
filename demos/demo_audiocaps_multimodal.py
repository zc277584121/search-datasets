#!/usr/bin/env python3
"""
AudioCaps - 多模态音频检索 Demo (CLAP)
========================================

场景说明:
- 使用 CLAP 模型同时编码音频和文本
- 支持：文本搜索音频、音频搜索相似音频
- 真正的多模态检索，直接处理音频波形

技术栈:
- Embedding: CLAP (laion/larger_clap_music_and_speech)
- Vector DB: Milvus Lite
- 音频处理: librosa
"""

import io
import sys
import numpy as np
import pandas as pd
import torch
import librosa
from pathlib import Path
from transformers import ClapProcessor, ClapModel
from pymilvus import MilvusClient

# Configuration
DATA_DIR = Path(__file__).parent.parent / "datasets" / "audio" / "audiocaps"
DB_PATH = str(Path(__file__).parent / "audiocaps_multimodal.db")
COLLECTION_NAME = "audiocaps_audio"
SAMPLE_SIZE = 200  # Number of audio clips to index
CLAP_MODEL = "laion/larger_clap_music_and_speech"
CLAP_DIM = 512
DEVICE = "cuda:1" if torch.cuda.is_available() else "cpu"  # Use GPU 1
SAMPLE_RATE = 48000  # CLAP expects 48kHz


class CLAPEmbedder:
    """CLAP model wrapper for audio and text embedding."""

    def __init__(self, model_name: str = CLAP_MODEL, device: str = DEVICE):
        print(f"加载 CLAP 模型: {model_name} (device: {device})")
        self.device = device
        self.model = ClapModel.from_pretrained(model_name).to(device)
        self.processor = ClapProcessor.from_pretrained(model_name)
        self.model.eval()

    @torch.no_grad()
    def encode_audios(self, audio_arrays: list, sample_rate: int = SAMPLE_RATE) -> list:
        """Encode a list of audio arrays to embeddings."""
        # Process audio - CLAP expects specific sample rate
        # Note: newer transformers use 'audio' instead of 'audios'
        inputs = self.processor(
            audio=audio_arrays,
            sampling_rate=sample_rate,
            return_tensors="pt",
            padding=True
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items() if isinstance(v, torch.Tensor)}
        outputs = self.model.get_audio_features(**inputs)
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


def load_audio_from_bytes(audio_bytes: bytes, target_sr: int = SAMPLE_RATE) -> np.ndarray:
    """Load audio from bytes and resample to target sample rate."""
    try:
        # Load audio from bytes
        audio_io = io.BytesIO(audio_bytes)
        waveform, sr = librosa.load(audio_io, sr=target_sr, mono=True)
        return waveform
    except Exception as e:
        print(f"音频加载失败: {e}")
        return None


def load_data():
    """Load AudioCaps data."""
    print("加载 AudioCaps 数据...")
    df = pd.read_parquet(DATA_DIR / "test.parquet")
    df_sample = df.head(SAMPLE_SIZE)
    print(f"加载了 {len(df_sample)} 条音频记录")
    return df_sample


def build_index(client: MilvusClient, embedder: CLAPEmbedder, df: pd.DataFrame):
    """Build Milvus index from audio clips."""
    print("\n正在构建音频索引...")

    # Create collection
    if client.has_collection(COLLECTION_NAME):
        client.drop_collection(COLLECTION_NAME)

    client.create_collection(
        collection_name=COLLECTION_NAME,
        dimension=CLAP_DIM,
        metric_type="COSINE",
        auto_id=True,
    )

    # Process audio in batches
    batch_size = 8  # Smaller batch for audio (more memory intensive)
    indexed_count = 0

    for i in range(0, len(df), batch_size):
        batch_df = df.iloc[i:i + batch_size]
        audios = []
        metadata = []

        for idx, (_, row) in enumerate(batch_df.iterrows()):
            context = row.get("context", {})
            if not isinstance(context, dict) or "bytes" not in context:
                continue

            audio_bytes = context["bytes"]
            waveform = load_audio_from_bytes(audio_bytes)
            if waveform is None:
                continue

            audios.append(waveform)
            caption = row.get("answer", "")
            instruction = row.get("instruction", "")

            metadata.append({
                "caption": str(caption)[:500],
                "instruction": str(instruction)[:200],
                "audio_id": str(i + idx),
            })

        if not audios:
            continue

        # Encode audios
        try:
            embeddings = embedder.encode_audios(audios)
        except Exception as e:
            print(f"  编码失败: {e}")
            continue

        # Insert to Milvus
        data = []
        for emb, meta in zip(embeddings, metadata):
            record = {"vector": emb}
            record.update(meta)
            data.append(record)

        client.insert(collection_name=COLLECTION_NAME, data=data)
        indexed_count += len(audios)
        print(f"  已索引 {indexed_count}/{len(df)} 条音频...")

    print(f"索引完成: {indexed_count} 条音频")
    return indexed_count


def search_by_text(client: MilvusClient, embedder: CLAPEmbedder, query: str, top_k: int = 5):
    """Search audio clips by text query."""
    print(f"\n文本搜索: {query}")

    # Encode query text
    query_embedding = embedder.encode_texts([query])[0]

    # Search
    results = client.search(
        collection_name=COLLECTION_NAME,
        data=[query_embedding],
        limit=top_k,
        output_fields=["caption", "instruction", "audio_id"]
    )

    print(f"\n{'='*60}")
    print(f"找到 {len(results[0])} 条相关音频:")
    print(f"{'='*60}")

    for i, result in enumerate(results[0], 1):
        similarity = result.get("distance", 0)
        entity = result.get("entity", result)
        caption = entity.get("caption", "N/A")
        audio_id = entity.get("audio_id", "N/A")

        print(f"\n[{i}] 相似度: {similarity:.4f} | ID: {audio_id}")
        print(f"    描述: {caption}")

    return results[0]


def search_by_audio_id(client: MilvusClient, embedder: CLAPEmbedder, df: pd.DataFrame,
                        audio_idx: int, top_k: int = 5):
    """Search similar audio clips by audio index."""
    print(f"\n音频搜索: 索引 {audio_idx}")

    # Get audio from dataframe
    row = df.iloc[audio_idx]
    context = row.get("context", {})
    if not isinstance(context, dict) or "bytes" not in context:
        print("无法获取音频数据")
        return []

    waveform = load_audio_from_bytes(context["bytes"])
    if waveform is None:
        print("无法加载音频")
        return []

    # Encode audio
    query_embedding = embedder.encode_audios([waveform])[0]

    # Search
    results = client.search(
        collection_name=COLLECTION_NAME,
        data=[query_embedding],
        limit=top_k + 1,
        output_fields=["caption", "instruction", "audio_id"]
    )

    print(f"\n{'='*60}")
    print(f"原始音频描述: {row.get('answer', 'N/A')}")
    print(f"找到 {len(results[0])} 条相似音频:")
    print(f"{'='*60}")

    for i, result in enumerate(results[0], 1):
        similarity = result.get("distance", 0)
        entity = result.get("entity", result)
        caption = entity.get("caption", "N/A")
        result_audio_id = entity.get("audio_id", "N/A")

        # Skip self
        if result_audio_id == str(audio_idx):
            continue

        print(f"\n[{i}] 相似度: {similarity:.4f} | ID: {result_audio_id}")
        print(f"    描述: {caption}")

    return results[0]


def main():
    print("=" * 60)
    print("AudioCaps 多模态音频检索 Demo (CLAP)")
    print("=" * 60)

    # Initialize
    embedder = CLAPEmbedder()
    milvus_client = MilvusClient(uri=DB_PATH)

    # Load and index data
    df = load_data()
    indexed = build_index(milvus_client, embedder, df)

    if indexed == 0:
        print("没有成功索引任何音频，退出")
        return

    # Demo: Text to Audio search
    text_queries = [
        "a dog barking loudly",
        "rain falling on the roof",
        "someone playing piano",
        "birds singing in the morning",
        "car engine starting",
    ]

    print("\n" + "=" * 60)
    print("文本 → 音频 搜索示例")
    print("=" * 60)

    for query in text_queries:
        search_by_text(milvus_client, embedder, query, top_k=3)

    # Demo: Audio to Audio search
    print("\n" + "=" * 60)
    print("音频 → 音频 搜索示例")
    print("=" * 60)

    # Search using first audio clip
    search_by_audio_id(milvus_client, embedder, df, audio_idx=0, top_k=3)

    # Interactive mode
    print("\n" + "=" * 60)
    print("交互模式")
    print("  输入声音描述进行搜索")
    print("  输入 'quit' 退出")
    print("=" * 60)

    while True:
        query = input("\n请输入声音描述: ").strip()
        if query.lower() in ["quit", "exit", "q"]:
            break
        if query:
            search_by_text(milvus_client, embedder, query)

    print("\nDemo 结束!")


if __name__ == "__main__":
    main()
