# FActScore dataset

[FActScore](https://github.com/shmsw25/FActScore) is a benchmark for fine-grained atomic evaluation of factual precision in long-form text generation. It evaluates whether the facts in a model-generated biography are supported by a knowledge source (Wikipedia).

## Create Hugging Face dataset

The Hugging Face dataset for FActScore can be found [here](https://huggingface.co/datasets/zypho/factscore). To reproduce this dataset, simply run the `create_huggingface_dataset.py` script.

## Metrics

The evaluation metrics are computed using the official FActScore evaluation script ([source](https://github.com/shmsw25/FActScore)), which is cloned under `thirdparty/factscore/`.

**Prerequisites**: FActScore evaluation requires downloading the Wikipedia knowledge source and setting an OpenAI API key. See the [FActScore repository](https://github.com/shmsw25/FActScore) for detailed setup instructions.

Metrics reported:
- **score**: FActScore (factual precision with length penalty)
- **init_score**: FActScore without length penalty
- **respond_ratio**: percentage of non-abstained responses
- **num_facts_per_response**: average number of atomic facts per valid response

## References

- [FActScore: Fine-grained Atomic Evaluation of Factual Precision in Long Form Text Generation](https://arxiv.org/abs/2305.14251)
- [FActScore GitHub repository](https://github.com/shmsw25/FActScore)
