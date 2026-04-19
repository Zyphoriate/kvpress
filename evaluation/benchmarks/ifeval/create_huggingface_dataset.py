# SPDX-FileCopyrightText: Copyright (c) 1993-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Script to convert the google/IFEval dataset to the kvpress standard format
# and push it to the Hugging Face Hub.
#
# Background
# ----------
# The raw google/IFEval dataset (https://huggingface.co/datasets/google/IFEval)
# has the following schema:
#
#   key               int
#   prompt            str     – the instruction prompt given to the model
#   instruction_id_list  list[str]  – IDs of the instructions embedded in the prompt
#   kwargs            list[dict]   – per-instruction keyword arguments for the checkers
#
# The kvpress evaluation framework expects datasets with:
#
#   context           str     – document/passage that is KV-cache compressed
#   question          str     – query appended after compression
#   answer_prefix     str     – prefix prepended to the model's answer
#   max_new_tokens    int     – maximum tokens to generate
#   answer            str     – reference answer (not applicable for IFEval)
#
# Additionally, the IFEval scorer needs:
#   instruction_id_list  list[str]  (preserved from source)
#   kwargs               list[dict] (preserved from source)
#
# Mapping
# -------
# Since IFEval has no long document to compress, the prompt itself is placed in
# `context` so that the full instruction is present when the KV cache is built.
# `question` is left empty (the instruction is already in context), and
# `answer_prefix` is also empty.
#
# Usage
# -----
#   cd evaluation/benchmarks/ifeval
#   python create_huggingface_dataset.py --hub_repo <your-hf-org/ifeval>

import argparse

from datasets import Dataset, load_dataset


def convert_ifeval_to_kvpress(hub_repo: str, hub_token: str | None = None) -> None:
    """Load google/IFEval, convert to kvpress format, and push to the Hub.

    Parameters
    ----------
    hub_repo:
        Hugging Face Hub repository in the form ``<org>/<dataset-name>``.
        Example: ``simonjegou/IFEval``.
    hub_token:
        Optional Hugging Face access token (write access required).  If
        ``None``, the token cached by ``huggingface-cli login`` is used.
    """
    print("Loading google/IFEval …")
    # google/IFEval only has a 'train' split (541 prompts)
    df = load_dataset("google/IFEval", split="train").to_pandas()

    # --- field mapping ---------------------------------------------------
    # The prompt is both the "context" (what the model attends to under KV
    # compression) and the sole input; there is no separate retrieval document.
    df["context"] = df["prompt"]
    df["question"] = ""
    df["answer_prefix"] = ""
    # Standard IFEval generation length used in the original paper.
    df["max_new_tokens"] = 1280
    # IFEval has no reference answer; the scorer operates on the response
    # directly via instruction checkers.
    df["answer"] = ""

    # Keep the columns needed by the scorer (instruction_id_list, kwargs)
    # together with the kvpress standard columns.
    columns = [
        "context",
        "question",
        "answer_prefix",
        "max_new_tokens",
        "answer",
        "key",
        "instruction_id_list",
        "kwargs",
    ]
    df = df[columns]

    dataset = Dataset.from_pandas(df, preserve_index=False)
    print(f"Converted dataset: {len(dataset)} samples")
    print(dataset.features)

    print(f"Pushing to Hub: {hub_repo} (split='test') …")
    # The converted dataset uses the standard kvpress split name 'test'.
    # After uploading, update evaluate_registry.py:
    #   DATASET_REGISTRY["ifeval"] = "<hub_repo>"
    # and remove the split='train' override from evaluate.py so that the
    # standard split='test' path is used.
    push_kwargs = {"repo_id": hub_repo, "split": "test"}
    if hub_token:
        push_kwargs["token"] = hub_token
    dataset.push_to_hub(**push_kwargs)
    print("Done.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Convert google/IFEval to kvpress format and push to the Hugging Face Hub."
    )
    parser.add_argument(
        "--hub_repo",
        required=True,
        help="HF Hub repository to push to, e.g. 'simonjegou/IFEval'.",
    )
    parser.add_argument(
        "--hub_token",
        default=None,
        help="HF access token (optional; falls back to cached login).",
    )
    args = parser.parse_args()
    convert_ifeval_to_kvpress(hub_repo=args.hub_repo, hub_token=args.hub_token)
