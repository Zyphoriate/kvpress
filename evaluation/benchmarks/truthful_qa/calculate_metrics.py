# SPDX-FileCopyrightText: Copyright (c) 1993-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import sys
from pathlib import Path

import pandas as pd

# Add the local thirdparty copy of truthfulqa to PYTHONPATH.
_thirdparty = Path(__file__).parent / "thirdparty"
if str(_thirdparty) not in sys.path:
    sys.path.insert(0, str(_thirdparty))

from truthfulqa import metrics  # noqa: E402

sys.path.pop(0)


def calculate_metrics(df: pd.DataFrame) -> dict:
    """
    Calculate TruthfulQA evaluation metrics using the official evaluation script.

    Metrics:
    - bleu_acc: BLEU-based accuracy (max correct > max incorrect)
    - rouge1_acc: ROUGE-1-based accuracy
    - rouge2_acc: ROUGE-2-based accuracy
    - rougeL_acc: ROUGE-L-based accuracy
    - bleurt_acc: BLEURT-based accuracy (max correct > max incorrect)
    """
    # Build a frame compatible with truthfulqa.metrics functions
    frame = pd.DataFrame({
        "Question": df["question"].apply(lambda x: x.replace("Q: ", "").replace("\n", "").strip() if isinstance(x, str) else ""),
        "Correct Answers": df["correct_answers"],
        "Incorrect Answers": df["incorrect_answers"],
    })
    frame["predicted_answer"] = df["predicted_answer"].apply(lambda x: x if isinstance(x, str) else "")

    # Run official BLEURT / BLEU / ROUGE metrics
    frame = metrics.run_bleu_and_rouge("predicted_answer", frame)
    frame = metrics.run_BLEURT("predicted_answer", frame)

    # Aggregate results
    metric_keys = {
        "bleu_acc": "predicted_answer bleu acc",
        "rouge1_acc": "predicted_answer rouge1 acc",
        "rouge2_acc": "predicted_answer rouge2 acc",
        "rougeL_acc": "predicted_answer rougeL acc",
        "bleurt_acc": "predicted_answer BLEURT acc",
    }

    result = {}
    for key, col in metric_keys.items():
        if col in frame.columns:
            result[key] = round(frame[col].mean(), 4)
        else:
            result[key] = 0.0

    result["total"] = len(df)
    return result
