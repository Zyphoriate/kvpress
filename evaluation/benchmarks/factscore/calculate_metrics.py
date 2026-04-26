# SPDX-FileCopyrightText: Copyright (c) 1993-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Calculate FActScore metrics using the official FActScore implementation."""

import logging
import os
import subprocess
import sys
import zipfile
from pathlib import Path

import pandas as pd

# Add the local thirdparty copy of FActScore to PYTHONPATH.
_thirdparty = Path(__file__).parent / "thirdparty"
if str(_thirdparty) not in sys.path:
    sys.path.insert(0, str(_thirdparty))

from factscore.factscorer import FactScorer  # noqa: E402

logging.basicConfig(
    format="%(asctime)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.CRITICAL,
)

# Google Drive file IDs for FActScore prerequisites
_ENWIKI_DB_ID = "1Qu4JHWjpUKhGPaAW5UHhS5RJ545CVy4I"
_DEMOS_ZIP_ID = "1sbW6pkYl6cc9gooD4WLaeoFKcAj3poZu"


def _find_llama_model_dir() -> str:
    """Locate a local Llama model under ``$HF_HOME/hub``."""
    hf_home = os.environ.get("HF_HOME", os.path.expanduser("~/.cache/huggingface"))
    hub_dir = Path(hf_home) / "hub"

    if not hub_dir.exists():
        raise FileNotFoundError(f"Hugging Face hub directory not found: {hub_dir}")

    for model_dir in hub_dir.glob("models--*llama*"):
        snapshots = model_dir / "snapshots"
        if snapshots.exists():
            for snapshot in snapshots.iterdir():
                if snapshot.is_dir() and (snapshot / "config.json").exists():
                    return str(snapshot)

    raise FileNotFoundError(
        f"No local Llama model found under {hub_dir}. "
        "Please cache a Llama-compatible model (e.g. meta-llama/Llama-2-7b-hf) first."
    )


def _download_file(file_id: str, dest: str) -> None:
    """Download a file from Google Drive via gdown or wget."""
    path = Path(dest)
    if path.exists():
        return

    try:
        import gdown

        gdown.download(id=file_id, output=str(path), quiet=False)
    except Exception:
        url = f"https://drive.google.com/uc?export=download&id={file_id}"
        subprocess.run(["wget", "-O", str(path), url], check=True)


def _ensure_factscore_prerequisites(cache_dir: Path) -> None:
    """Download Wikipedia DB and demos if they are not already cached."""
    cache_dir.mkdir(parents=True, exist_ok=True)

    # Wikipedia knowledge source database
    db_path = cache_dir / "enwiki-20230401.db"
    if not db_path.exists():
        logging.critical("Downloading enwiki-20230401.db (this may take a while)...")
        _download_file(_ENWIKI_DB_ID, str(db_path))

    # Demos for atomic fact generation
    demos_dir = cache_dir / "demos"
    if not demos_dir.exists():
        demos_zip = cache_dir / "demos.zip"
        logging.critical("Downloading demos.zip ...")
        _download_file(_DEMOS_ZIP_ID, str(demos_zip))
        with zipfile.ZipFile(demos_zip, "r") as zf:
            zf.extractall(cache_dir)
        demos_zip.unlink()


def calculate_metrics(df: pd.DataFrame) -> dict:
    """Calculate FActScore evaluation metrics.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing at least ``topic`` and ``predicted_answer`` columns.

    Returns
    -------
    dict
        Dictionary with ``score``, ``init_score``, ``respond_ratio``,
        ``num_facts_per_response``, and ``total`` keys.
    """
    topics = []
    generations = []

    for _, row in df.iterrows():
        topic = row.get("topic", "")
        if not isinstance(topic, str) or not topic.strip():
            # Fallback: try to extract topic from the prompt context.
            context = row.get("context", "")
            if isinstance(context, str) and "bio of " in context:
                topic = context.split("bio of ")[-1].rstrip(".")
            else:
                topic = ""
        topics.append(topic.strip())

        pred = row.get("predicted_answer", "")
        generations.append(pred if isinstance(pred, str) else "")

    # Filter out completely empty generations to avoid evaluation errors.
    valid_indices = [i for i, gen in enumerate(generations) if gen.strip()]
    if not valid_indices:
        return {
            "score": 0.0,
            "init_score": 0.0,
            "respond_ratio": 0.0,
            "num_facts_per_response": 0.0,
            "total": len(df),
        }

    valid_topics = [topics[i] for i in valid_indices]
    valid_generations = [generations[i] for i in valid_indices]

    model_dir = _find_llama_model_dir()
    factscore_cache = Path(__file__).parent / ".cache"
    _ensure_factscore_prerequisites(factscore_cache)

    fs = FactScorer(
        model_name="retrieval+llama+npm",
        data_dir=str(factscore_cache),
        model_dir=model_dir,
        cache_dir=str(factscore_cache),
    )

    out = fs.get_score(
        topics=valid_topics,
        generations=valid_generations,
        gamma=10,
        verbose=True,
    )

    metrics = {
        "score": round(float(out.get("score", 0.0)), 4),
        "respond_ratio": round(float(out.get("respond_ratio", 0.0)), 4),
        "num_facts_per_response": round(float(out.get("num_facts_per_response", 0.0)), 4),
        "total": len(df),
    }

    if "init_score" in out:
        metrics["init_score"] = round(float(out["init_score"]), 4)

    return metrics
