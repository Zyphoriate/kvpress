# TruthfulQA dataset

[TruthfulQA](https://github.com/sylinrl/TruthfulQA) is a benchmark to measure whether a language model is truthful in generating answers to questions. The benchmark comprises 817 questions that span 38 categories, including health, law, finance and politics.

## Create Hugging Face dataset

The Hugging Face dataset for TruthfulQA can be found [here](https://huggingface.co/datasets/zzyppp/truthful_qa). To reproduce this dataset, simply run the `create_huggingface_dataset.py` script.

## Metrics

The evaluation metrics are computed using the official TruthfulQA evaluation script ([source](https://github.com/sylinrl/TruthfulQA/tree/main/truthfulqa)), which is cloned under `thirdparty/truthfulqa/`.

Metrics reported:
- **bleu_acc**: BLEU-based accuracy (max correct > max incorrect)
- **rouge1_acc**: ROUGE-1-based accuracy
- **rouge2_acc**: ROUGE-2-based accuracy
- **rougeL_acc**: ROUGE-L-based accuracy
- **bleurt_acc**: BLEURT-based accuracy (max correct > max incorrect)

## References

- [TruthfulQA: Measuring How Models Mimic Human Falsehoods](https://arxiv.org/abs/2109.07958)
- [TruthfulQA GitHub repository](https://github.com/sylinrl/TruthfulQA)
- [TruthfulQA Hugging Face dataset](https://huggingface.co/datasets/domenicrosati/TruthfulQA)
