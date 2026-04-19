# SPDX-FileCopyrightText: Copyright (c) 1993-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import os
import sys

import nltk
import pandas as pd

# Add the third_party directory to sys.path so that `instruction_following_eval`
# is importable as a package.  The insertion is guarded to be idempotent and
# only adds a single vendored directory.  Sub-modules within the package use
# absolute `instruction_following_eval.*` imports which require the package root
# to be on sys.path; importlib-based loading is therefore impractical without
# also mutating sys.path.
_THIRD_PARTY_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "third_party")
if _THIRD_PARTY_DIR not in sys.path:
    sys.path.insert(0, _THIRD_PARTY_DIR)

from instruction_following_eval.evaluation_lib import InputExample, compute_metrics  # noqa: E402
from instruction_following_eval.evaluation_lib import (  # noqa: E402
    test_instruction_following_loose,
    test_instruction_following_strict,
)


def _ensure_nltk_data():
    """Download required NLTK data if not already present."""
    resources = [
        ("tokenizers/punkt_tab", "punkt_tab"),
        ("taggers/averaged_perceptron_tagger_eng", "averaged_perceptron_tagger_eng"),
    ]
    for find_path, download_name in resources:
        try:
            nltk.data.find(find_path)
        except LookupError:
            nltk.download(download_name, quiet=True)


def calculate_metrics(df: pd.DataFrame) -> dict:
    """Compute IFEval strict and loose prompt-level / instruction-level accuracy.

    Each row in *df* must contain:
    - ``context`` (or ``prompt``): the original instruction prompt.
    - ``predicted_answer``: model-generated text.
    - ``instruction_id_list``: list of instruction ID strings.
    - ``kwargs``: list of dicts, one per instruction.

    Returns
    -------
    dict with keys:
        ``strict_prompt_level_accuracy``       – strict: fraction of prompts where ALL instructions are satisfied.
        ``strict_instruction_level_accuracy``  – strict: fraction of individual instructions satisfied.
        ``loose_prompt_level_accuracy``        – loose:  same, with minor response variations.
        ``loose_instruction_level_accuracy``   – loose:  same, with minor response variations.
        ``per_category_accuracy_strict``       – dict of per-category strict accuracy.
        ``per_category_accuracy_loose``        – dict of per-category loose accuracy.
        ``num_prompts``                        – total evaluated prompts.
        ``num_instructions``                   – total evaluated instructions.
    """
    _ensure_nltk_data()

    # Build InputExample list and prompt→response mapping
    inputs = []
    prompt_to_response = {}

    # Support both 'prompt' (raw google/IFEval) and 'context' (kvpress-mapped) columns
    prompt_col = "prompt" if "prompt" in df.columns else "context"

    for _, row in df.iterrows():
        prompt = row[prompt_col]
        response = row["predicted_answer"]
        if not isinstance(response, str):
            response = ""

        instruction_ids = row["instruction_id_list"]
        kwargs_list = row["kwargs"]
        # Ensure kwargs entries are dicts (may be loaded as None from CSV)
        kwargs_list = [kw if isinstance(kw, dict) else {} for kw in kwargs_list]

        inputs.append(
            InputExample(
                key=int(row.get("key", 0)),
                instruction_id_list=list(instruction_ids),
                prompt=prompt,
                kwargs=kwargs_list,
            )
        )
        prompt_to_response[prompt] = response

    # Run strict and loose evaluation using the original google evaluation_lib
    strict_outputs = [test_instruction_following_strict(inp, prompt_to_response) for inp in inputs]
    loose_outputs = [test_instruction_following_loose(inp, prompt_to_response) for inp in inputs]

    strict_metrics = compute_metrics(strict_outputs)
    loose_metrics = compute_metrics(loose_outputs)

    return {
        "strict_prompt_level_accuracy": strict_metrics["prompt_level_accuracy"],
        "strict_instruction_level_accuracy": strict_metrics["instruction_level_accuracy"],
        "loose_prompt_level_accuracy": loose_metrics["prompt_level_accuracy"],
        "loose_instruction_level_accuracy": loose_metrics["instruction_level_accuracy"],
        "per_category_accuracy_strict": strict_metrics["per_category_accuracy"],
        "per_category_accuracy_loose": loose_metrics["per_category_accuracy"],
        "num_prompts": strict_metrics["num_prompts"],
        "num_instructions": strict_metrics["num_instructions"],
    }
