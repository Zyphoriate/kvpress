# SPDX-FileCopyrightText: Copyright (c) 1993-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Calculate FActScore metrics using the official FActScore implementation."""

import logging
import os
import re
import shutil
import sqlite3
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

# Add the local thirdparty copy of FActScore to PYTHONPATH.
_thirdparty = Path(__file__).parent / "thirdparty"
if str(_thirdparty) not in sys.path:
    sys.path.insert(0, str(_thirdparty))

from factscore.download_data import download_file  # noqa: E402
from factscore.factscorer import FactScorer  # noqa: E402

logging.basicConfig(
    format="%(asctime)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.CRITICAL,
)

# Google Drive file IDs for FActScore prerequisites
_ENWIKI_DB_ID = "1Qu4JHWjpUKhGPaAW5UHhS5RJ545CVy4I"
_DEMOS_ZIP_ID = "1sbW6pkYl6cc9gooD4WLaeoFKcAj3poZu"


def _configure_deepseek(cache_dir: Path) -> str:
    """Set up DeepSeek API as an OpenAI-compatible backend.

    Returns the path to a temporary API key file that FActScore can read.
    """
    api_key = os.environ.get("DEEPSEEK_API_KEY", "")
    api_base = os.environ.get("DEEPSEEK_API_BASE", "https://api.deepseek.com")

    # Write key to disk — OpenAIModel.load_model() reads from a file
    key_file = cache_dir / "deepseek_api.key"
    key_file.write_text(api_key)

    # Configure openai globals (works for both old and new openai library versions)
    import openai as oa

    oa.api_key = api_key
    try:
        oa.api_base = api_base
    except AttributeError:
        # openai >= 1.0 renamed api_base → base_url (handled below)
        pass

    # Monkey-patch model names and API shapes for DeepSeek compatibility
    from factscore import openai_lm as oal

    _orig_chat = oal.call_ChatGPT

    def _patched_chat(message, model_name="deepseek-chat", max_len=1024, temp=0.7, verbose=False):
        return _orig_chat(message, model_name=model_name, max_len=max_len, temp=temp, verbose=verbose)

    oal.call_ChatGPT = _patched_chat

    # DeepSeek only supports Chat Completions, so replace GPT-3 Completion calls
    def _patched_gpt3(prompt, model_name="deepseek-chat", max_len=512, temp=0.7,
                       num_log_probs=0, echo=False, verbose=False):
        response = None
        received = False
        num_rate_errors = 0
        while not received:
            try:
                response = oa.ChatCompletion.create(
                    model=model_name,
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=max_len,
                    temperature=temp,
                )
                received = True
            except Exception:
                error = sys.exc_info()[0]
                num_rate_errors += 1
                logging.error("API error: %s (%d)", error, num_rate_errors)
                time.sleep(np.power(2, num_rate_errors))
        return response

    oal.call_GPT3 = _patched_gpt3

    return str(key_file)


def _download_large_file(file_id: str, dest: Path) -> None:
    """Download a large file from Google Drive, handling the virus-scan confirmation page.

    Uses ``gdown`` when available, otherwise falls back to a ``requests``-based
    approach that extracts confirm / uuid tokens from the warning page.
    """
    # --- try gdown first (handles all of this automatically) ---
    try:
        import gdown

        gdown.download(id=file_id, output=str(dest), quiet=False)
        return
    except Exception:
        pass

    # --- requests-based fallback ---
    import requests

    base_url = "https://docs.google.com/uc?export=download&id=" + file_id
    session = requests.Session()

    # Step 1: get the intermediate page (may be a warning for large files)
    resp = session.get(base_url)
    resp.raise_for_status()

    # Step 2: parse confirm token + uuid from the virus-warning form if present
    confirm = "t"  # default
    match = re.search(r'name="confirm"\s+value="(.*?)"', resp.text)
    if match:
        confirm = match.group(1)

    uuid = None
    match = re.search(r'name="uuid"\s+value="(.*?)"', resp.text)
    if match:
        uuid = match.group(1)

    # Step 3: build the final download URL
    download_url = (
        f"https://drive.usercontent.google.com/download"
        f"?id={file_id}&export=download&confirm={confirm}"
    )
    if uuid:
        download_url += f"&uuid={uuid}"

    # Step 4: stream the actual file to disk
    resp = session.get(download_url, stream=True)
    resp.raise_for_status()

    total = int(resp.headers.get("content-length", 0))
    with open(dest, "wb") as f:
        downloaded = 0
        for chunk in resp.iter_content(chunk_size=8 * 1024 * 1024):
            f.write(chunk)  # type: ignore[arg-type]
            downloaded += len(chunk)
            if total:
                pct = downloaded / total * 100
                logging.info(f"Downloading {dest.name}: {downloaded // 1024**2} / {total // 1024**2} MB ({pct:.0f}%)")


def _is_valid_db(db_path: Path) -> bool:
    """Check whether the file is a valid SQLite database (not an HTML warning page)."""
    try:
        conn = sqlite3.connect(str(db_path))
        cursor = conn.cursor()
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
        result = cursor.fetchall()
        conn.close()
        return len(result) > 0
    except Exception:
        return False


def _ensure_factscore_prerequisites(cache_dir: Path) -> None:
    """Download Wikipedia DB, demos, and stopwords if they are not already cached."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_dir_str = str(cache_dir)

    # Wikipedia knowledge source database (20 GB) — gdown handles large file confirmation
    db_path = cache_dir / "enwiki-20230401.db"
    if not db_path.exists() or (db_path.exists() and not _is_valid_db(db_path)):
        if db_path.exists():
            logging.critical("Removing invalid enwiki-20230401.db (was HTML warning page)...")
            db_path.unlink()
        logging.critical("Downloading enwiki-20230401.db (this may take a while)...")
        _download_large_file(_ENWIKI_DB_ID, db_path)

    # Demos for atomic fact generation (small file, download_file handles zips)
    demos_dir = cache_dir / "demos"
    if not demos_dir.exists():
        logging.critical("Downloading demos.zip ...")
        download_file(_DEMOS_ZIP_ID, str(cache_dir / "demos.zip"), cache_dir_str)

    # roberta_stopwords.txt is required by npm.py and read from the working directory
    stopwords_src = _thirdparty / "roberta_stopwords.txt"
    if stopwords_src.exists():
        stopwords_dst = cache_dir / "roberta_stopwords.txt"
        if not stopwords_dst.exists():
            shutil.copy2(stopwords_src, stopwords_dst)
        # npm.py reads from cwd, so also place a copy there
        cwd_copy = Path("roberta_stopwords.txt")
        if not cwd_copy.exists():
            shutil.copy2(stopwords_src, cwd_copy)

    # spaCy model required by atomic_facts.py
    try:
        import spacy

        spacy.load("en_core_web_sm")
    except OSError:
        logging.critical("Downloading spaCy model en_core_web_sm ...")
        import subprocess

        subprocess.run(
            [sys.executable, "-m", "spacy", "download", "en_core_web_sm"], check=True
        )


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

    factscore_cache = Path(__file__).parent / ".cache"
    _ensure_factscore_prerequisites(factscore_cache)

    openai_key_path = _configure_deepseek(factscore_cache)

    fs = FactScorer(
        model_name="retrieval+ChatGPT",
        openai_key=openai_key_path,
        data_dir=str(factscore_cache),
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
