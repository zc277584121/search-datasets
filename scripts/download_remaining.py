#!/usr/bin/env python3
"""
Download FinQA and CUAD from their original sources (GitHub).
"""

import os
import json
import logging
import zipfile
import requests
from pathlib import Path

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

BASE_DIR = Path(__file__).parent.parent / "datasets"


def download_finqa():
    """Download FinQA from GitHub."""
    logger.info("=" * 60)
    logger.info("Downloading FinQA from GitHub...")
    save_dir = BASE_DIR / "finance" / "finqa"
    save_dir.mkdir(parents=True, exist_ok=True)

    # FinQA GitHub raw URLs
    base_url = "https://raw.githubusercontent.com/czyssrs/FinQA/main/dataset"
    files = {
        "train": f"{base_url}/train.json",
        "dev": f"{base_url}/dev.json",
        "test": f"{base_url}/test.json",
    }

    try:
        info = {"name": "FinQA", "splits": {}, "status": "success"}

        for split_name, url in files.items():
            logger.info(f"  Downloading {split_name}...")
            response = requests.get(url, timeout=60)
            response.raise_for_status()
            data = response.json()

            # Save as JSON
            json_path = save_dir / f"{split_name}.json"
            with open(json_path, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2, ensure_ascii=False)

            info["splits"][split_name] = {"num_rows": len(data), "file": str(json_path)}
            logger.info(f"    - {split_name}: {len(data)} examples")

        # Get sample structure for documentation
        if "train" in files:
            sample = data[0] if data else {}
            info["sample_keys"] = list(sample.keys()) if sample else []

        with open(save_dir / "metadata.json", "w", encoding="utf-8") as f:
            json.dump(info, f, indent=2, ensure_ascii=False)

        logger.info("✓ FinQA downloaded successfully!")
        return True

    except Exception as e:
        logger.error(f"✗ FinQA failed: {e}")
        return False


def download_cuad():
    """Download CUAD from GitHub."""
    logger.info("=" * 60)
    logger.info("Downloading CUAD from GitHub...")
    save_dir = BASE_DIR / "document" / "cuad"
    save_dir.mkdir(parents=True, exist_ok=True)

    # CUAD uses SQuAD format JSON files
    # The data is in the releases
    try:
        # Download from the GitHub release
        zip_url = "https://github.com/TheAtticusProject/cuad/raw/main/data.zip"
        logger.info("  Downloading data.zip...")

        response = requests.get(zip_url, timeout=120, stream=True)
        response.raise_for_status()

        zip_path = save_dir / "data.zip"
        with open(zip_path, "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)

        logger.info("  Extracting...")
        with zipfile.ZipFile(zip_path, "r") as z:
            z.extractall(save_dir)

        # Remove zip file
        zip_path.unlink()

        # Count data
        info = {"name": "CUAD", "splits": {}, "status": "success"}

        # Check for extracted JSON files
        for json_file in save_dir.glob("**/*.json"):
            if json_file.name in ["train_separate_questions.json", "test.json"]:
                with open(json_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                split_name = json_file.stem
                # SQuAD format has 'data' key
                if "data" in data:
                    num_questions = sum(
                        len(para.get("qas", []))
                        for article in data["data"]
                        for para in article.get("paragraphs", [])
                    )
                    info["splits"][split_name] = {"num_questions": num_questions}
                    logger.info(f"    - {split_name}: {num_questions} questions")

        with open(save_dir / "metadata.json", "w", encoding="utf-8") as f:
            json.dump(info, f, indent=2, ensure_ascii=False)

        logger.info("✓ CUAD downloaded successfully!")
        return True

    except Exception as e:
        logger.error(f"✗ CUAD failed: {e}")
        return False


def main():
    logger.info("Downloading remaining datasets...")
    results = {
        "FinQA": download_finqa(),
        "CUAD": download_cuad(),
    }

    logger.info("\n" + "=" * 60)
    logger.info("SUMMARY")
    for name, success in results.items():
        status = "✓" if success else "✗"
        logger.info(f"  {status} {name}")


if __name__ == "__main__":
    main()
