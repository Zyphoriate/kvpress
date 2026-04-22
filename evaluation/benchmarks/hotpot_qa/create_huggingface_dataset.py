# SPDX-FileCopyrightText: Copyright (c) 1993-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0


from datasets import Dataset, load_dataset

# Templates adapted from LongBench hotpotqa formatting
context_prefix = (
    "Answer the question based on the given passages. Only give me the answer and do not output any other words.\n\n"
    "The following are given passages.\n{context}\n\n"
    "Answer the question based on the given passages. Only give me the answer and do not output any other words.\n\n"
)

question_template = "Question: {input}\n"
answer_prefix = "Answer:"
max_new_tokens = 52


def format_context(example):
    passages = []
    for i, (title, sentences) in enumerate(
        zip(example["context"]["title"], example["context"]["sentences"]), 1
    ):
        passage_text = " ".join(sentences)
        passages.append(f"Passage {i}:\n{title}\n{passage_text}")

    context_body = "\n\n".join(passages)
    return {"context": context_prefix.format(context=context_body)}


# HotpotQA distractor setting only has train/validation splits with answers.
# We use the validation split as the test split for evaluation.
dataset = load_dataset("hotpotqa/hotpot_qa", "distractor", split="validation")
dataset = dataset.map(format_context)
dataset = dataset.map(lambda x: {"question": question_template.format(input=x["question"])})
dataset = dataset.map(lambda x: {"answer_prefix": answer_prefix})
dataset = dataset.map(lambda x: {"task": "hotpotqa"})
dataset = dataset.map(lambda x: {"max_new_tokens": max_new_tokens})
# HotpotQA uses a single string answer; wrap it in a list for consistency with other benchmarks
dataset = dataset.map(lambda x: {"answers": [x["answer"]]})

df = dataset.to_pandas()
df = df[["context", "question", "answer_prefix", "answers", "task", "max_new_tokens"]]

# Push to hub as test split
processed_dataset = Dataset.from_pandas(df)
processed_dataset.push_to_hub("zzyppp/hotpot_qa", config_name="distractor", split="test")
