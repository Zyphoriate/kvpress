# SPDX-FileCopyrightText: Copyright (c) 1993-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Create the Hugging Face dataset for FActScore.

FActScore data is downloaded from the original Google Drive release,
processed into the standard evaluation format, and pushed to the hub.
"""

import subprocess
import zipfile
from pathlib import Path

from datasets import Dataset

# Google Drive file ID for FActScore data.zip
DATA_ZIP_ID = "155exEdKs7R21gZF4G-x54-XN3qswBcPo"
CACHE_DIR = Path(__file__).parent / ".cache"
DATA_ZIP_PATH = CACHE_DIR / "data.zip"


def _download_data():
    """Download and extract FActScore data if not already present."""
    CACHE_DIR.mkdir(parents=True, exist_ok=True)

    if not DATA_ZIP_PATH.exists():
        try:
            import gdown

            gdown.download(id=DATA_ZIP_ID, output=str(DATA_ZIP_PATH), quiet=False)
        except Exception:
            # Fallback to wget if gdown is unavailable
            url = f"https://drive.google.com/uc?export=download&id={DATA_ZIP_ID}"
            subprocess.run(["wget", "-O", str(DATA_ZIP_PATH), url], check=True)

    extracted_dir = CACHE_DIR / "data"
    if not extracted_dir.exists():
        with zipfile.ZipFile(DATA_ZIP_PATH, "r") as zf:
            zf.extractall(CACHE_DIR)


def _load_entities(split: str = "unlabeled") -> list[str]:
    """Load entity names from the requested split."""
    entities_path = CACHE_DIR / "data" / split / "prompt_entities.txt"
    with open(entities_path, encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]


def main():
    _download_data()

    # Use the unlabeled split (500 entities) as the evaluation set.
    entities = _load_entities("unlabeled")

    context_template = "Tell me a bio of {topic}."
    max_new_tokens = 256

    records = []
    for topic in entities:
        records.append(
            {
                "context": context_template.format(topic=topic),
                "question": "",
                "answer_prefix": "",
                "answer": "",
                "task": "factscore",
                "topic": topic,
                "max_new_tokens": max_new_tokens,
            }
        )

    dataset = Dataset.from_list(records)
    dataset.push_to_hub("zypho/factscore", split="test")


if __name__ == "__main__":
    main()
