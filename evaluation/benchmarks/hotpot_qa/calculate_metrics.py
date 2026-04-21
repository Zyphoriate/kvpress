# SPDX-FileCopyrightText: Copyright (c) 1993-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json
import sys
from pathlib import Path

import pandas as pd

# Add the thirdparty hotpot directory to the path so we can import the original evaluation code
_thirdparty_dir = Path(__file__).parent / "thirdparty" / "hotpot"
sys.path.insert(0, str(_thirdparty_dir))

# Monkey-patch ujson with standard json in case ujson is not installed
sys.modules.setdefault("ujson", json)

from hotpot_evaluate_v1 import exact_match_score, f1_score

sys.path.pop(0)


def calculate_metrics(df: pd.DataFrame) -> dict:
    """
    Calculate HotpotQA evaluation metrics using the official evaluation script.

    Metrics:
    - em: Exact Match
    - f1: F1 score
    - prec: Precision
    - recall: Recall
    """
    metrics = {"em": 0.0, "f1": 0.0, "prec": 0.0, "recall": 0.0}

    for _, row in df.iterrows():
        prediction = row["predicted_answer"] if isinstance(row["predicted_answer"], str) else ""
        # answers is a list with a single string for hotpotqa
        ground_truths = row["answers"]
        if isinstance(ground_truths, str):
            ground_truths = [ground_truths]

        # HotpotQA uses a single answer, but we handle list for consistency with other benchmarks
        gold = ground_truths[0] if len(ground_truths) > 0 else ""

        em = exact_match_score(prediction, gold)
        f1, prec, recall = f1_score(prediction, gold)

        metrics["em"] += float(em)
        metrics["f1"] += f1
        metrics["prec"] += prec
        metrics["recall"] += recall

    n = len(df)
    for k in metrics.keys():
        metrics[k] /= n

    # Round for readability
    metrics = {k: round(v, 4) for k, v in metrics.items()}
    metrics["total"] = n

    return metrics
