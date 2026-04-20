# SPDX-FileCopyrightText: Copyright (c) 1993-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Adapted from google-research/instruction_following_eval
# (commit aa633e5105c702b47a4dd836d9b6eca39984a0fe)
# Original license: Apache-2.0
# Changes: replaced `instruction_following_eval.*` imports with local imports.

"""Evaluation library for instruction following."""

import collections
import dataclasses
from typing import Dict, List, Optional, Union

from instruction_following_eval import instructions_registry


@dataclasses.dataclass
class InputExample:
    key: int
    instruction_id_list: List[str]
    prompt: str
    kwargs: List[Dict[str, Optional[Union[str, int]]]]


@dataclasses.dataclass
class OutputExample:
    instruction_id_list: List[str]
    prompt: str
    response: str
    follow_all_instructions: bool
    follow_instruction_list: List[bool]


def test_instruction_following_strict(inp, prompt_to_response):
    """Tests response to see if instructions are followed (strict mode)."""
    response = prompt_to_response[inp.prompt]
    instruction_list = inp.instruction_id_list
    is_following_list = []

    for index, instruction_id in enumerate(instruction_list):
        instruction_cls = instructions_registry.INSTRUCTION_DICT[instruction_id]
        instruction = instruction_cls(instruction_id)

        instruction.build_description(**inp.kwargs[index])
        args = instruction.get_instruction_args()
        if args and "prompt" in args:
            instruction.build_description(prompt=inp.prompt)

        if response.strip() and instruction.check_following(response):
            is_following_list.append(True)
        else:
            is_following_list.append(False)

    return OutputExample(
        instruction_id_list=inp.instruction_id_list,
        prompt=inp.prompt,
        response=response,
        follow_all_instructions=all(is_following_list),
        follow_instruction_list=is_following_list,
    )


def test_instruction_following_loose(inp, prompt_to_response):
    """Tests response for an upper bound for following instructions (loose mode)."""
    response = prompt_to_response[inp.prompt]
    r = response.split("\n")
    response_remove_first = "\n".join(r[1:]).strip()
    response_remove_last = "\n".join(r[:-1]).strip()
    response_remove_both = "\n".join(r[1:-1]).strip()
    revised_response = response.replace("*", "")
    revised_response_remove_first = response_remove_first.replace("*", "")
    revised_response_remove_last = response_remove_last.replace("*", "")
    revised_response_remove_both = response_remove_both.replace("*", "")
    all_responses = [
        response,
        revised_response,
        response_remove_first,
        response_remove_last,
        response_remove_both,
        revised_response_remove_first,
        revised_response_remove_last,
        revised_response_remove_both,
    ]
    instruction_list = inp.instruction_id_list
    is_following_list = []

    for index, instruction_id in enumerate(instruction_list):
        instruction_cls = instructions_registry.INSTRUCTION_DICT[instruction_id]
        instruction = instruction_cls(instruction_id)

        instruction.build_description(**inp.kwargs[index])
        args = instruction.get_instruction_args()
        if args and "prompt" in args:
            instruction.build_description(prompt=inp.prompt)

        is_following = False
        for r in all_responses:
            if r.strip() and instruction.check_following(r):
                is_following = True
                break

        is_following_list.append(is_following)

    return OutputExample(
        instruction_id_list=inp.instruction_id_list,
        prompt=inp.prompt,
        response=response,
        follow_all_instructions=all(is_following_list),
        follow_instruction_list=is_following_list,
    )


def compute_metrics(outputs):
    """Compute aggregated accuracy metrics from a list of OutputExample.

    Returns a dict with:
    - prompt_level_accuracy
    - instruction_level_accuracy
    - per_category_accuracy  (dict keyed by instruction category)
    - num_prompts
    - num_instructions
    """
    prompt_total = 0
    prompt_correct = 0
    instruction_total = 0
    instruction_correct = 0

    tier0_total: Dict[str, int] = collections.defaultdict(int)
    tier0_correct: Dict[str, int] = collections.defaultdict(int)

    for example in outputs:
        follow_instruction_list = example.follow_instruction_list
        instruction_id_list = example.instruction_id_list

        prompt_total += 1
        if all(follow_instruction_list):
            prompt_correct += 1

        instruction_total += len(instruction_id_list)
        instruction_correct += sum(follow_instruction_list)

        for instruction_id, followed_or_not in zip(instruction_id_list, follow_instruction_list):
            category = instruction_id.split(":")[0]
            tier0_total[category] += 1
            if followed_or_not:
                tier0_correct[category] += 1

    per_category = {
        cat: round(tier0_correct[cat] / tier0_total[cat], 4) for cat in sorted(tier0_total.keys())
    }

    return {
        "prompt_level_accuracy": round(prompt_correct / prompt_total, 4) if prompt_total > 0 else 0.0,
        "instruction_level_accuracy": (
            round(instruction_correct / instruction_total, 4) if instruction_total > 0 else 0.0
        ),
        "per_category_accuracy": per_category,
        "num_prompts": prompt_total,
        "num_instructions": instruction_total,
    }
