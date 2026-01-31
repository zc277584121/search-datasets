#!/usr/bin/env python3
"""
Download remaining failed datasets with alternative methods.
"""

import json
import logging
import requests
from pathlib import Path
from datasets import load_dataset

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

BASE_DIR = Path(__file__).parent.parent / "datasets"


def download_enron_alternative():
    """Download Enron using alternative dataset."""
    logger.info("Downloading Enron (alternative: SetFit/enron_spam)...")
    save_dir = BASE_DIR / "document" / "enron_mini"
    save_dir.mkdir(parents=True, exist_ok=True)

    try:
        # Use SetFit enron_spam which is smaller and cleaner
        dataset = load_dataset("SetFit/enron_spam")

        info = {"name": "Enron Spam Email", "hf_path": "SetFit/enron_spam", "splits": {}, "status": "success"}

        for split_name, split_data in dataset.items():
            split_path = save_dir / f"{split_name}.parquet"
            split_data.to_parquet(str(split_path))
            info["splits"][split_name] = {"num_rows": len(split_data)}
            logger.info(f"  - {split_name}: {len(split_data)} rows")

        info["features"] = {k: str(v) for k, v in dataset[list(dataset.keys())[0]].features.items()}

        with open(save_dir / "metadata.json", "w", encoding="utf-8") as f:
            json.dump(info, f, indent=2, ensure_ascii=False)

        logger.info("✓ Enron downloaded!")
        return True
    except Exception as e:
        logger.error(f"✗ Enron failed: {e}")
        return False


def download_spreadsheet_alternative():
    """Download SpreadsheetBench instructions only."""
    logger.info("Downloading SpreadsheetBench (instructions JSON)...")
    save_dir = BASE_DIR / "table" / "spreadsheet_bench"
    save_dir.mkdir(parents=True, exist_ok=True)

    try:
        # Download just the instructions JSON from GitHub
        url = "https://raw.githubusercontent.com/RUCKBReasworking/SpreadsheetBench/main/data/spreadsheetbench_912_v0.1/instructions.jsonl"
        response = requests.get(url, timeout=60)

        if response.status_code == 200:
            # Save as JSONL
            jsonl_path = save_dir / "instructions.jsonl"
            with open(jsonl_path, "w", encoding="utf-8") as f:
                f.write(response.text)

            # Count lines
            lines = response.text.strip().split('\n')
            num_rows = len(lines)

            info = {
                "name": "SpreadsheetBench",
                "source": "GitHub",
                "splits": {"train": {"num_rows": num_rows}},
                "status": "success",
                "note": "Instructions only, Excel files need to be downloaded separately"
            }

            with open(save_dir / "metadata.json", "w", encoding="utf-8") as f:
                json.dump(info, f, indent=2, ensure_ascii=False)

            logger.info(f"✓ SpreadsheetBench downloaded! ({num_rows} instructions)")
            return True
        else:
            # Try alternative: just create a placeholder with info
            logger.warning("Could not download from GitHub, trying HF viewer...")
            # The dataset structure is complex, let's just note it
            info = {
                "name": "SpreadsheetBench",
                "hf_path": "KAKA22/SpreadsheetBench",
                "status": "partial",
                "note": "Complex TAR format - use HF viewer or download manually",
                "manual_download": "https://huggingface.co/datasets/KAKA22/SpreadsheetBench"
            }
            with open(save_dir / "metadata.json", "w", encoding="utf-8") as f:
                json.dump(info, f, indent=2, ensure_ascii=False)
            return False

    except Exception as e:
        logger.error(f"✗ SpreadsheetBench failed: {e}")
        return False


def main():
    logger.info("Downloading remaining datasets...")

    results = {
        "Enron": download_enron_alternative(),
        "SpreadsheetBench": download_spreadsheet_alternative(),
    }

    logger.info(f"\n{'='*60}")
    logger.info("SUMMARY")
    for name, success in results.items():
        status = "✓" if success else "✗"
        logger.info(f"  {status} {name}")


if __name__ == "__main__":
    main()
