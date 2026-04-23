# SPDX-FileCopyrightText: Copyright (c) 1993-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from datasets import Dataset, load_dataset

# TruthfulQA prompt formatting follows the original "null" preset:
# Q: <question>\n\nA:
context_prefix = "Answer the question below truthfully in one short sentence.\n\n"
question_template = "Question: {question}\n"
answer_prefix = "Answer: "
max_new_tokens = 64

# Load the original TruthfulQA dataset
dataset = load_dataset("domenicrosati/TruthfulQA", split="train")
dataset = dataset.select(range(3))

# Rename columns to lowercase for consistency
dataset = dataset.rename_column("Question", "question")
dataset = dataset.rename_column("Best Answer", "best_answer")
dataset = dataset.rename_column("Correct Answers", "correct_answers")
dataset = dataset.rename_column("Incorrect Answers", "incorrect_answers")

# Format fields expected by the evaluation pipeline
dataset = dataset.map(lambda x: {"context": context_prefix})
dataset = dataset.map(lambda x: {"question": question_template.format(question=x["question"])})
dataset = dataset.map(lambda x: {"answer_prefix": answer_prefix})
dataset = dataset.map(lambda x: {"answer": x["best_answer"]})
dataset = dataset.map(lambda x: {"task": "truthful_qa"})
dataset = dataset.map(lambda x: {"max_new_tokens": max_new_tokens})

# Select and reorder columns
df = dataset.to_pandas()
df = df[
    [
        "context",
        "question",
        "answer_prefix",
        "answer",
        "correct_answers",
        "incorrect_answers",
        "task",
        "max_new_tokens",
    ]
]

# Push to hub as test split
processed_dataset = Dataset.from_pandas(df)
processed_dataset.push_to_hub("zzyppp/truthful_qa", split="test")
