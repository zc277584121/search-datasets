#!/usr/bin/env python3
"""
Download 6 small interesting datasets.
"""

import json
import logging
from pathlib import Path
from datasets import load_dataset

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

BASE_DIR = Path(__file__).parent.parent / "datasets"

DATASETS = [
    {
        "name": "WildChat Creative Writing 10K",
        "hf_path": "sam-paech/wildchat_creative_writing_annotated_10k",
        "save_dir": BASE_DIR / "conversation" / "wildchat_10k",
    },
    {
        "name": "Discord Chat",
        "hf_path": "breadlicker45/discord-chat",
        "save_dir": BASE_DIR / "conversation" / "discord_chat",
    },
    {
        "name": "ELI5 Reddit",
        "hf_path": "Pavithree/eli5",
        "save_dir": BASE_DIR / "conversation" / "eli5_reddit",
    },
    {
        "name": "GitHub Issues",
        "hf_path": "lewtun/github-issues",
        "save_dir": BASE_DIR / "code" / "github_issues",
    },
    {
        "name": "Enron Email Mini",
        "hf_path": "amanneo/enron-mail-corpus-mini",
        "save_dir": BASE_DIR / "document" / "enron_mini",
    },
    {
        "name": "SpreadsheetBench",
        "hf_path": "KAKA22/SpreadsheetBench",
        "save_dir": BASE_DIR / "table" / "spreadsheet_bench",
    },
]


def download_dataset(config: dict) -> dict:
    """Download a single dataset."""
    name = config["name"]
    hf_path = config["hf_path"]
    save_dir = config["save_dir"]

    logger.info(f"{'='*60}")
    logger.info(f"Downloading: {name}")
    logger.info(f"From: {hf_path}")

    save_dir.mkdir(parents=True, exist_ok=True)

    try:
        dataset = load_dataset(hf_path)

        info = {"name": name, "hf_path": hf_path, "splits": {}, "status": "success"}

        for split_name, split_data in dataset.items():
            split_path = save_dir / f"{split_name}.parquet"
            split_data.to_parquet(str(split_path))
            info["splits"][split_name] = {"num_rows": len(split_data)}
            logger.info(f"  - {split_name}: {len(split_data)} rows")

        # Get features
        first_split = list(dataset.keys())[0]
        info["features"] = {k: str(v) for k, v in dataset[first_split].features.items()}

        with open(save_dir / "metadata.json", "w", encoding="utf-8") as f:
            json.dump(info, f, indent=2, ensure_ascii=False)

        logger.info(f"✓ {name} downloaded!")
        return info

    except Exception as e:
        logger.error(f"✗ {name} failed: {e}")
        return {"name": name, "status": "failed", "error": str(e)}


def main():
    logger.info("Downloading 6 small datasets...")

    results = []
    for config in DATASETS:
        result = download_dataset(config)
        results.append(result)

    # Summary
    logger.info(f"\n{'='*60}")
    logger.info("SUMMARY")
    success = [r for r in results if r.get("status") == "success"]
    failed = [r for r in results if r.get("status") == "failed"]

    logger.info(f"Success: {len(success)}/{len(DATASETS)}")
    for r in success:
        total = sum(s["num_rows"] for s in r.get("splits", {}).values())
        logger.info(f"  ✓ {r['name']}: {total} rows")

    if failed:
        logger.info(f"Failed: {len(failed)}")
        for r in failed:
            logger.info(f"  ✗ {r['name']}: {r.get('error')}")


if __name__ == "__main__":
    main()
