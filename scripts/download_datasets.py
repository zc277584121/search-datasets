#!/usr/bin/env python3
"""
Download script for the 10 selected datasets.
Each dataset will be saved to its designated directory.
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

# Dataset configurations
DATASETS = [
    {
        "name": "SQuAD 2.0",
        "hf_path": "rajpurkar/squad_v2",
        "save_dir": BASE_DIR / "text" / "squad2",
        "description": "Stanford Question Answering Dataset v2.0"
    },
    {
        "name": "CMRC 2018",
        "hf_path": "hfl/cmrc2018",
        "save_dir": BASE_DIR / "chinese" / "cmrc2018",
        "description": "Chinese Machine Reading Comprehension 2018"
    },
    {
        "name": "FinQA",
        "hf_path": "dreamerdeo/finqa",
        "save_dir": BASE_DIR / "finance" / "finqa",
        "description": "Financial Question Answering with Numerical Reasoning"
    },
    {
        "name": "CUAD",
        "hf_path": "theatticusproject/cuad-qa",
        "save_dir": BASE_DIR / "document" / "cuad",
        "description": "Contract Understanding Atticus Dataset"
    },
    {
        "name": "Spider",
        "hf_path": "xlangai/spider",
        "save_dir": BASE_DIR / "table" / "spider",
        "description": "Cross-domain Text-to-SQL Dataset"
    },
    {
        "name": "ChartQA",
        "hf_path": "HuggingFaceM4/ChartQA",
        "save_dir": BASE_DIR / "multimodal" / "chartqa",
        "description": "Chart Question Answering Dataset"
    },
    {
        "name": "COCO Karpathy 5K",
        "hf_path": "nlphuji/mscoco_2014_5k_test_image_text_retrieval",
        "save_dir": BASE_DIR / "multimodal" / "coco_karpathy",
        "description": "COCO Karpathy 5K Test Split for Image-Text Retrieval"
    },
    {
        "name": "MSVD",
        "hf_path": "friedrichor/MSVD",
        "save_dir": BASE_DIR / "video" / "msvd",
        "description": "Microsoft Video Description Dataset"
    },
    {
        "name": "AudioCaps Test",
        "hf_path": "AudioLLMs/audiocaps_test",
        "save_dir": BASE_DIR / "audio" / "audiocaps",
        "description": "AudioCaps Test Set for Audio Captioning"
    },
    {
        "name": "MultiHop-RAG",
        "hf_path": "yixuantt/MultiHopRAG",
        "save_dir": BASE_DIR / "rag" / "multihop_rag",
        "description": "Multi-hop Retrieval-Augmented Generation Dataset"
    },
]


def download_dataset(config: dict) -> dict:
    """Download a single dataset and return its info."""
    name = config["name"]
    hf_path = config["hf_path"]
    save_dir = config["save_dir"]

    logger.info(f"=" * 60)
    logger.info(f"Downloading: {name}")
    logger.info(f"HuggingFace path: {hf_path}")
    logger.info(f"Save directory: {save_dir}")

    # Ensure directory exists
    save_dir.mkdir(parents=True, exist_ok=True)

    try:
        # Load dataset
        dataset = load_dataset(hf_path, trust_remote_code=True)

        # Get dataset info
        info = {
            "name": name,
            "hf_path": hf_path,
            "splits": {},
            "features": None,
            "status": "success"
        }

        # Process each split
        for split_name, split_data in dataset.items():
            split_path = save_dir / f"{split_name}.parquet"
            split_data.to_parquet(str(split_path))

            info["splits"][split_name] = {
                "num_rows": len(split_data),
                "file": str(split_path)
            }
            logger.info(f"  - {split_name}: {len(split_data)} rows -> {split_path}")

        # Get features (column info)
        first_split = list(dataset.keys())[0]
        info["features"] = {
            k: str(v) for k, v in dataset[first_split].features.items()
        }

        # Save metadata
        meta_path = save_dir / "metadata.json"
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(info, f, indent=2, ensure_ascii=False)

        logger.info(f"✓ {name} downloaded successfully!")
        return info

    except Exception as e:
        logger.error(f"✗ Failed to download {name}: {e}")
        return {
            "name": name,
            "hf_path": hf_path,
            "status": "failed",
            "error": str(e)
        }


def main():
    """Download all datasets."""
    logger.info("Starting dataset download...")
    logger.info(f"Total datasets to download: {len(DATASETS)}")

    results = []
    for i, config in enumerate(DATASETS, 1):
        logger.info(f"\n[{i}/{len(DATASETS)}] Processing {config['name']}...")
        result = download_dataset(config)
        results.append(result)

    # Summary
    logger.info("\n" + "=" * 60)
    logger.info("DOWNLOAD SUMMARY")
    logger.info("=" * 60)

    success = [r for r in results if r.get("status") == "success"]
    failed = [r for r in results if r.get("status") == "failed"]

    logger.info(f"Successful: {len(success)}/{len(DATASETS)}")
    for r in success:
        total_rows = sum(s["num_rows"] for s in r.get("splits", {}).values())
        logger.info(f"  ✓ {r['name']}: {total_rows} total rows")

    if failed:
        logger.info(f"\nFailed: {len(failed)}/{len(DATASETS)}")
        for r in failed:
            logger.info(f"  ✗ {r['name']}: {r.get('error', 'Unknown error')}")

    # Save overall summary
    summary_path = BASE_DIR / "download_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    logger.info(f"\nSummary saved to: {summary_path}")

    logger.info("\nDownload complete!")


if __name__ == "__main__":
    main()
