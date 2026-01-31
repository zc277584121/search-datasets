#!/usr/bin/env python3
"""
Download script for the 4 failed datasets with correct parameters.
"""

import os
import json
import logging
from pathlib import Path
from datasets import load_dataset

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Base directory
BASE_DIR = Path(__file__).parent.parent / "datasets"


def download_finqa():
    """Download FinQA dataset."""
    logger.info("=" * 60)
    logger.info("Downloading FinQA...")
    save_dir = BASE_DIR / "finance" / "finqa"
    save_dir.mkdir(parents=True, exist_ok=True)

    try:
        # Try alternative repo or direct loading
        dataset = load_dataset("ibm-research/finqa")

        info = {"name": "FinQA", "splits": {}, "status": "success"}

        for split_name, split_data in dataset.items():
            split_path = save_dir / f"{split_name}.parquet"
            split_data.to_parquet(str(split_path))
            info["splits"][split_name] = {"num_rows": len(split_data)}
            logger.info(f"  - {split_name}: {len(split_data)} rows")

        info["features"] = {k: str(v) for k, v in dataset[list(dataset.keys())[0]].features.items()}

        with open(save_dir / "metadata.json", "w", encoding="utf-8") as f:
            json.dump(info, f, indent=2, ensure_ascii=False)

        logger.info("✓ FinQA downloaded successfully!")
        return True
    except Exception as e:
        logger.error(f"✗ FinQA failed: {e}")
        return False


def download_cuad():
    """Download CUAD dataset."""
    logger.info("=" * 60)
    logger.info("Downloading CUAD...")
    save_dir = BASE_DIR / "document" / "cuad"
    save_dir.mkdir(parents=True, exist_ok=True)

    try:
        # Use the standard cuad dataset
        dataset = load_dataset("cuad")

        info = {"name": "CUAD", "splits": {}, "status": "success"}

        for split_name, split_data in dataset.items():
            split_path = save_dir / f"{split_name}.parquet"
            split_data.to_parquet(str(split_path))
            info["splits"][split_name] = {"num_rows": len(split_data)}
            logger.info(f"  - {split_name}: {len(split_data)} rows")

        info["features"] = {k: str(v) for k, v in dataset[list(dataset.keys())[0]].features.items()}

        with open(save_dir / "metadata.json", "w", encoding="utf-8") as f:
            json.dump(info, f, indent=2, ensure_ascii=False)

        logger.info("✓ CUAD downloaded successfully!")
        return True
    except Exception as e:
        logger.error(f"✗ CUAD failed: {e}")
        return False


def download_coco_karpathy():
    """Download COCO Karpathy 5K dataset."""
    logger.info("=" * 60)
    logger.info("Downloading COCO Karpathy 5K...")
    save_dir = BASE_DIR / "multimodal" / "coco_karpathy"
    save_dir.mkdir(parents=True, exist_ok=True)

    try:
        # Try yerevann/coco-karpathy which is a standard parquet dataset
        dataset = load_dataset("yerevann/coco-karpathy")

        info = {"name": "COCO Karpathy", "splits": {}, "status": "success"}

        for split_name, split_data in dataset.items():
            split_path = save_dir / f"{split_name}.parquet"
            split_data.to_parquet(str(split_path))
            info["splits"][split_name] = {"num_rows": len(split_data)}
            logger.info(f"  - {split_name}: {len(split_data)} rows")

        info["features"] = {k: str(v) for k, v in dataset[list(dataset.keys())[0]].features.items()}

        with open(save_dir / "metadata.json", "w", encoding="utf-8") as f:
            json.dump(info, f, indent=2, ensure_ascii=False)

        logger.info("✓ COCO Karpathy downloaded successfully!")
        return True
    except Exception as e:
        logger.error(f"✗ COCO Karpathy failed: {e}")
        return False


def download_multihop_rag():
    """Download MultiHop-RAG dataset."""
    logger.info("=" * 60)
    logger.info("Downloading MultiHop-RAG...")
    save_dir = BASE_DIR / "rag" / "multihop_rag"
    save_dir.mkdir(parents=True, exist_ok=True)

    try:
        # Load both configs
        queries = load_dataset("yixuantt/MultiHopRAG", "MultiHopRAG")
        corpus = load_dataset("yixuantt/MultiHopRAG", "corpus")

        info = {"name": "MultiHop-RAG", "splits": {}, "status": "success"}

        # Save queries
        for split_name, split_data in queries.items():
            split_path = save_dir / f"queries_{split_name}.parquet"
            split_data.to_parquet(str(split_path))
            info["splits"][f"queries_{split_name}"] = {"num_rows": len(split_data)}
            logger.info(f"  - queries_{split_name}: {len(split_data)} rows")

        # Save corpus
        for split_name, split_data in corpus.items():
            split_path = save_dir / f"corpus_{split_name}.parquet"
            split_data.to_parquet(str(split_path))
            info["splits"][f"corpus_{split_name}"] = {"num_rows": len(split_data)}
            logger.info(f"  - corpus_{split_name}: {len(split_data)} rows")

        # Get features from queries
        info["query_features"] = {k: str(v) for k, v in queries[list(queries.keys())[0]].features.items()}
        info["corpus_features"] = {k: str(v) for k, v in corpus[list(corpus.keys())[0]].features.items()}

        with open(save_dir / "metadata.json", "w", encoding="utf-8") as f:
            json.dump(info, f, indent=2, ensure_ascii=False)

        logger.info("✓ MultiHop-RAG downloaded successfully!")
        return True
    except Exception as e:
        logger.error(f"✗ MultiHop-RAG failed: {e}")
        return False


def main():
    """Download all failed datasets."""
    logger.info("Downloading failed datasets...")

    results = {
        "FinQA": download_finqa(),
        "CUAD": download_cuad(),
        "COCO Karpathy": download_coco_karpathy(),
        "MultiHop-RAG": download_multihop_rag(),
    }

    logger.info("\n" + "=" * 60)
    logger.info("SUMMARY")
    logger.info("=" * 60)
    for name, success in results.items():
        status = "✓" if success else "✗"
        logger.info(f"  {status} {name}")

    logger.info("\nDone!")


if __name__ == "__main__":
    main()
