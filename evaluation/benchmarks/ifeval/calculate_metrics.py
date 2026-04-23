# SPDX-FileCopyrightText: Copyright (c) 1993-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0


import sys
import pandas as pd
from pathlib import Path


# Add the local third_party copy of instruction_following_eval to PYTHONPATH.
_third_party = Path(__file__).parent / "third_party"
if str(_third_party) not in sys.path:
    sys.path.insert(0, str(_third_party))

from instruction_following_eval import evaluation_lib  # noqa: E402

INPUT_DATA_PATH = f"{_third_party}/instruction_following_eval/data/input_data.jsonl"


def calculate_metrics(df: pd.DataFrame) -> dict:
    prompt_to_response = {
        row["question"]: row["predicted_answer"]
        if isinstance(row["predicted_answer"], str)
        else ""
        for _, row in df.iterrows()
    }

    inputs = evaluation_lib.read_prompt_list(INPUT_DATA_PATH)
    inputs = [inp for inp in inputs if inp.prompt in prompt_to_response.keys()]

    metrics = {}
    for func, key in [
        (evaluation_lib.test_instruction_following_strict, "strict_accuracy"),
        (evaluation_lib.test_instruction_following_loose, "loose_accuracy"),
    ]:
        outputs = [func(inp, prompt_to_response) for inp in inputs]
        follow_all = [o.follow_all_instructions for o in outputs]
        metrics[key] = round(sum(follow_all) / len(outputs), 4) if outputs else 0.0

    metrics["total"] = len(df)
    return metrics
