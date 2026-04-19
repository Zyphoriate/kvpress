# SPDX-FileCopyrightText: Copyright (c) 1993-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pandas as pd

from benchmarks.ifeval.instructions import check_instruction


def calculate_metrics(df: pd.DataFrame) -> dict:
    """Compute IFEval prompt-level and instruction-level accuracy.

    Each row in *df* must contain:
    - ``predicted_answer``: model-generated text.
    - ``instruction_id_list``: list of instruction IDs (strings).
    - ``kwargs``: list of dicts, one per instruction.

    Returns
    -------
    dict
        ``prompt_level_accuracy``   – fraction of prompts where ALL instructions
                                      are satisfied.
        ``instruction_level_accuracy`` – fraction of individual instructions that
                                         are satisfied.
        ``num_prompts``             – total number of evaluated prompts.
        ``num_instructions``        – total number of evaluated instructions.
    """
    prompt_correct = 0
    instruction_correct = 0
    total_instructions = 0

    for _, row in df.iterrows():
        response = row["predicted_answer"]
        if not isinstance(response, str):
            response = ""

        instruction_ids = row["instruction_id_list"]
        kwargs_list = row["kwargs"]

        all_followed = True
        for instr_id, kw in zip(instruction_ids, kwargs_list):
            followed = check_instruction(instr_id, response, kw if isinstance(kw, dict) else {})
            instruction_correct += int(followed)
            total_instructions += 1
            if not followed:
                all_followed = False

        prompt_correct += int(all_followed)

    n = len(df)
    return {
        "prompt_level_accuracy": round(prompt_correct / n, 4) if n > 0 else 0.0,
        "instruction_level_accuracy": (
            round(instruction_correct / total_instructions, 4) if total_instructions > 0 else 0.0
        ),
        "num_prompts": n,
        "num_instructions": total_instructions,
    }
