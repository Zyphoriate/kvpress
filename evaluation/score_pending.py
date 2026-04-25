# SPDX-FileCopyrightText: Copyright (c) 1993-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Script to score pending evaluations in the results directory.

This script scans the results directory for folders that have predictions.csv
but no metrics.json, calculates metrics using the appropriate scorer, and saves
the results.

Usage:
    python score_pending.py --results_dir ./results
"""

import json
import logging
import re
from pathlib import Path
from typing import Optional

import pandas as pd
import yaml
from fire import Fire

from evaluate_registry import SCORER_REGISTRY

logger = logging.getLogger(__name__)


def parse_dir_name(dir_name: str) -> Optional[dict]:
    """
    Parse the results directory name to extract dataset, model, press, and compression ratio.

    Expected format: {dataset}__[{data_dir}__]{model}__{press}__{compression_ratio}
    Model may contain '--' as separator (e.g., 'Qwen--Qwen3-8B').
    data_dir is optional and only present for some datasets (e.g., loogle).

    Parameters
    ----------
    dir_name : str
        The directory name to parse.

    Returns
    -------
    dict or None
        Dictionary with keys: dataset, model, press, compression_ratio, extra_info
        Returns None if parsing fails.
    """
    parts = dir_name.split("__")

    if len(parts) < 4:
        logger.warning(f"Directory name '{dir_name}' does not match expected format.")
        return None

    # Try to match the dataset name from the beginning.
    # Some dataset names contain underscores (e.g., "hotpot_qa", "needle_in_haystack"),
    # so we try known datasets from longest to shortest to avoid partial matches.
    known_datasets = sorted(SCORER_REGISTRY.keys(), key=len, reverse=True)
    dataset = None
    data_dir = None
    for ds in known_datasets:
        ds_parts = ds.split("_")
        if parts[: len(ds_parts)] == ds_parts:
            dataset = ds
            remaining = parts[len(ds_parts) :]
            break
    else:
        # Fallback: assume the first part is the dataset name
        dataset = parts[0]
        remaining = parts[1:]

    if len(remaining) < 3:
        logger.warning(f"Directory name '{dir_name}' does not have enough parts after dataset.")
        return None

    # Extract extra suffixes from the last part (e.g., "0.10__fraction0.500")
    extra_info = {}
    last_part = remaining[-1]
    if "__" in last_part:
        cr_parts = last_part.split("__")
        remaining[-1] = cr_parts[0]
        for extra in cr_parts[1:]:
            match = re.match(r"([a-zA-Z_]+)([0-9.]+)", extra)
            if match:
                key, value = match.groups()
                try:
                    value = float(value) if "." in value else int(value)
                except ValueError:
                    pass
                extra_info[key] = value

    compression_ratio = remaining[-1]
    press = remaining[-2]
    # Model is everything between dataset and press, excluding data_dir if present.
    # Model names always contain '--' (original '/' replaced).
    model_parts = remaining[:-2]
    # If the first model part does not contain '--', it might be a data_dir.
    # We identify data_dir as parts without '--' before the actual model name.
    while model_parts and "--" not in model_parts[0]:
        data_dir = model_parts.pop(0) if data_dir is None else f"{data_dir}_{model_parts.pop(0)}"

    model = "__".join(model_parts).replace("--", "/")

    return {
        "dataset": dataset,
        "data_dir": data_dir,
        "model": model,
        "press": press,
        "compression_ratio": float(compression_ratio) if "." in compression_ratio else int(compression_ratio),
        "extra_info": extra_info,
    }


def process_directory(results_dir: Path, dir_path: Path) -> bool:
    """
    Process a single results directory: calculate metrics if only predictions.csv exists.

    Parameters
    ----------
    results_dir : Path
        The root results directory.
    dir_path : Path
        The specific subdirectory to process.

    Returns
    -------
    bool
        True if metrics were calculated and saved, False otherwise.
    """
    dir_name = dir_path.name

    predictions_file = dir_path / "predictions.csv"
    metrics_file = dir_path / "metrics.json"
    config_file = dir_path / "config.yaml"

    # Check if predictions.csv exists
    if not predictions_file.exists():
        logger.debug(f"Skipping {dir_name}: no predictions.csv found.")
        return False

    # Check if metrics.json already exists
    if metrics_file.exists():
        logger.debug(f"Skipping {dir_name}: metrics.json already exists.")
        return False

    # Parse directory name to get dataset info
    config = parse_dir_name(dir_name)
    if config is None:
        logger.warning(f"Could not parse directory name: {dir_name}")
        return False

    dataset = config["dataset"]

    # Check if we have a scorer for this dataset
    if dataset not in SCORER_REGISTRY:
        logger.warning(f"No scorer found for dataset '{dataset}' in {dir_name}")
        return False

    scorer = SCORER_REGISTRY[dataset]

    logger.info(f"Processing {dir_name} (dataset: {dataset})")

    try:
        # Load predictions
        df = pd.read_csv(predictions_file)
        logger.info(f"Loaded {len(df)} predictions from {predictions_file}")

        # Calculate metrics
        logger.info(f"Calculating metrics using {scorer.__module__}")
        metrics = scorer(df)

        # Save metrics
        with open(metrics_file, "w") as f:
            json.dump(metrics, f, indent=4)
        logger.info(f"Saved metrics to {metrics_file}")

        # Save config if it doesn't exist
        if not config_file.exists():
            config_data = {
                "dataset": dataset,
                "model": config["model"],
                "press_name": config["press"],
                "compression_ratio": config["compression_ratio"],
            }
            # Add extra info (fraction, max_context, etc.)
            config_data.update(config["extra_info"])

            with open(config_file, "w") as f:
                yaml.dump(config_data, f, default_flow_style=False, indent=2, sort_keys=False)
            logger.info(f"Saved config to {config_file}")

        return True

    except Exception as e:
        logger.error(f"Failed to process {dir_name}: {e}", exc_info=True)
        return False


def score_pending_results(results_dir: str = "./results", log_level: str = "INFO"):
    """
    Scan results directory and calculate metrics for pending evaluations.

    A pending evaluation is a subdirectory that contains predictions.csv but no metrics.json.

    Parameters
    ----------
    results_dir : str
        Path to the results directory. Default is "./results".
    log_level : str
        Logging level (DEBUG, INFO, WARNING, ERROR). Default is "INFO".
    """
    # Setup logging
    logging.basicConfig(
        level=log_level.upper(),
        format="%(asctime)s - %(levelname)s - %(message)s",
    )

    results_path = Path(results_dir)

    if not results_path.exists():
        logger.error(f"Results directory does not exist: {results_path}")
        return

    logger.info(f"Scanning {results_path} for pending evaluations...")

    processed = 0
    skipped = 0
    failed = 0

    for dir_path in results_path.iterdir():
        if not dir_path.is_dir():
            continue

        predictions_file = dir_path / "predictions.csv"
        metrics_file = dir_path / "metrics.json"

        if not predictions_file.exists():
            skipped += 1
            continue

        if metrics_file.exists():
            skipped += 1
            continue

        # This is a pending evaluation
        logger.info(f"Found pending evaluation: {dir_path.name}")

        if process_directory(results_path, dir_path):
            processed += 1
        else:
            failed += 1

    logger.info("=" * 60)
    logger.info(f"Scan complete: {processed} processed, {skipped} skipped, {failed} failed")


if __name__ == "__main__":
    Fire(score_pending_results)
