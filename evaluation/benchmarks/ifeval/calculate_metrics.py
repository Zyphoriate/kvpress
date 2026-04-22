"""IFEval metrics using Google's official instruction_following_eval implementation."""

# SPDX-FileCopyrightText: Copyright (c) 1993-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import ast
import importlib
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import pandas as pd


def _get_evaluation_lib():
        third_party_dir = Path(__file__).resolve().parent / "third_party"
        if str(third_party_dir) not in sys.path:
                sys.path.insert(0, str(third_party_dir))

        return importlib.import_module("instruction_following_eval.evaluation_lib")


def _parse_maybe_serialized(value: Any) -> Any:
        if isinstance(value, str):
                stripped = value.strip()
                if not stripped:
                        return value
                for parser in (json.loads, ast.literal_eval):
                        try:
                                return parser(stripped)
                        except Exception:
                                continue
        return value


def _coerce_instruction_ids(value: Any) -> list[str]:
        parsed = _parse_maybe_serialized(value)
        if isinstance(parsed, list):
                return [str(item) for item in parsed]
        raise TypeError(f"instruction_id_list should be a list, got {type(parsed)}")


def _coerce_kwargs(value: Any, expected_len: int) -> list[dict[str, Any]]:
        parsed = _parse_maybe_serialized(value)
        if isinstance(parsed, list):
                kwargs = [item if isinstance(item, dict) else {} for item in parsed]
        else:
                kwargs = []
        if len(kwargs) < expected_len:
                kwargs.extend({} for _ in range(expected_len - len(kwargs)))
        return kwargs[:expected_len]


def _build_inputs(df: pd.DataFrame) -> tuple[list[Any], dict[str, str]]:
        eval_lib = _get_evaluation_lib()
        inputs: list[Any] = []
        prompt_to_response: dict[str, str] = {}

        required_columns = {"prompt", "instruction_id_list", "kwargs", "predicted_answer"}
        missing = required_columns.difference(df.columns)
        if missing:
                raise ValueError(f"Missing required columns for IFEval: {sorted(missing)}")

        for row_idx, row in df.reset_index(drop=True).iterrows():
                prompt = str(row["prompt"])
                instruction_ids = _coerce_instruction_ids(row["instruction_id_list"])
                kwargs = _coerce_kwargs(row["kwargs"], expected_len=len(instruction_ids))
                key = int(row["key"]) if "key" in df.columns else row_idx

                inputs.append(
                        eval_lib.InputExample(
                                key=key,
                                instruction_id_list=instruction_ids,
                                prompt=prompt,
                                kwargs=kwargs,
                        )
                )

                prediction = row["predicted_answer"]
                if pd.isna(prediction):
                        prediction = ""
                prompt_to_response[prompt] = str(prediction)

        return inputs, prompt_to_response


def _summarize_outputs(outputs: list[Any]) -> dict[str, Any]:
        prompt_total = len(outputs)
        prompt_correct = sum(1 for out in outputs if out.follow_all_instructions)
        instruction_total = sum(len(out.follow_instruction_list) for out in outputs)
        instruction_correct = sum(sum(out.follow_instruction_list) for out in outputs)

        tier0_total: dict[str, int] = defaultdict(int)
        tier0_correct: dict[str, int] = defaultdict(int)
        tier1_total: dict[str, int] = defaultdict(int)
        tier1_correct: dict[str, int] = defaultdict(int)

        for out in outputs:
                for instruction_id, followed in zip(out.instruction_id_list, out.follow_instruction_list):
                        tier0_id = instruction_id.split(":")[0]
                        tier0_total[tier0_id] += 1
                        tier1_total[instruction_id] += 1
                        if followed:
                                tier0_correct[tier0_id] += 1
                                tier1_correct[instruction_id] += 1

        per_instruction_tier0 = {
                instruction_id: tier0_correct[instruction_id] / total
                for instruction_id, total in sorted(tier0_total.items())
                if total > 0
        }
        per_instruction_tier1 = {
                instruction_id: tier1_correct[instruction_id] / total
                for instruction_id, total in sorted(tier1_total.items())
                if total > 0
        }

        return {
                "prompt_total": prompt_total,
                "prompt_correct": prompt_correct,
                "prompt_accuracy": prompt_correct / prompt_total if prompt_total else 0.0,
                "instruction_total": instruction_total,
                "instruction_correct": instruction_correct,
                "instruction_accuracy": instruction_correct / instruction_total if instruction_total else 0.0,
                "per_instruction_tier0": per_instruction_tier0,
                "per_instruction_tier1": per_instruction_tier1,
        }


def calculate_metrics(df: pd.DataFrame) -> dict[str, Any]:
        eval_lib = _get_evaluation_lib()
        inputs, prompt_to_response = _build_inputs(df)

        strict_outputs = [eval_lib.test_instruction_following_strict(inp, prompt_to_response) for inp in inputs]
        loose_outputs = [eval_lib.test_instruction_following_loose(inp, prompt_to_response) for inp in inputs]

        return {
                "official_google_eval": True,
                "num_examples": len(inputs),
                "strict": _summarize_outputs(strict_outputs),
                "loose": _summarize_outputs(loose_outputs),
        }