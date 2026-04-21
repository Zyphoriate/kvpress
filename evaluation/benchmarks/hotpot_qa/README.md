# HotpotQA dataset

[HotpotQA](https://hotpotqa.github.io/) is a dataset for diverse, explainable multi-hop question answering.

## Create Hugging Face dataset

The Hugging Face dataset for HotpotQA can be found [here](https://huggingface.co/datasets/simonjegou/hotpot_qa). To reproduce this dataset, simply run the `create_huggingface_dataset.py` script.

## Metrics

The evaluation metrics are computed using the official HotpotQA evaluation script ([source](https://github.com/hotpotqa/hotpot/blob/master/hotpot_evaluate_v1.py)), which is cloned under `thirdparty/hotpot/`.

Metrics reported:
- **em**: Exact Match
- **f1**: F1 score
- **prec**: Precision
- **recall**: Recall

## References

- [HotpotQA: A Dataset for Diverse, Explainable Multi-hop Question Answering](https://arxiv.org/abs/1809.09600)
- [HotpotQA GitHub repository](https://github.com/hotpotqa/hotpot)
- [HotpotQA Hugging Face dataset](https://huggingface.co/datasets/hotpotqa/hotpot_qa)
