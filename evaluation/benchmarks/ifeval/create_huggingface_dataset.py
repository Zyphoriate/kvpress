# SPDX-FileCopyrightText: Copyright (c) 1993-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from datasets import Dataset, load_dataset

# IFEval does not have a train/test split; use the full dataset as test.
context_prefix = "Strictly follow the instruction and complete the task: {prompt}"
question_prefix = ""
answer_prefix = "Response: "
max_new_tokens = 1280


# IFEval does not have a train/test split; use the full dataset as test.
dataset = load_dataset("google/IFEval", split="train")
dataset = dataset.map(lambda x: {"context": context_prefix})
dataset = dataset.map(lambda x: {"question": question_prefix.format(prompt=x["prompt"])})
dataset = dataset.map(lambda x: {"answer_prefix": answer_prefix})
dataset = dataset.map(lambda x: {"answer": ""})
dataset = dataset.map(lambda x: {"task": "ifeval"})
dataset = dataset.map(lambda x: {"max_new_tokens": max_new_tokens})

df = dataset.to_pandas()
df = df[["context", "question", "answer_prefix", "answer", "task", "prompt", "max_new_tokens"]] # type: ignore

# Push to hub as test split
processed_dataset = Dataset.from_pandas(df)
processed_dataset.push_to_hub("zypho/ifeval", split="test")