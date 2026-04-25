# SPDX-FileCopyrightText: Copyright (c) 1993-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from datasets import Dataset, load_dataset

# TruthfulQA prompt formatting follows the original "null" preset:
# Q: <question>\n\nA:
context_prefix = "Based on your knowledge, answer the question in one short sentence: {question}"
question_template = ""
answer_prefix = "Answer: "
max_new_tokens = 64

# Load the original TruthfulQA dataset
dataset = load_dataset("domenicrosati/TruthfulQA", split="train")

# Rename columns to lowercase for consistency
dataset = dataset.rename_column("Question", "question")
dataset = dataset.rename_column("Best Answer", "best_answer")
dataset = dataset.rename_column("Correct Answers", "correct_answers")
dataset = dataset.rename_column("Incorrect Answers", "incorrect_answers")

# Format fields expected by the evaluation pipeline
dataset = dataset.map(lambda x: {"context": context_prefix.format(question=x["question"])})
dataset = dataset.map(lambda x: {"question": question_template})
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
