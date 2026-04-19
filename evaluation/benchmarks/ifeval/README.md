# IFEval benchmark

[IFEval](https://arxiv.org/abs/2311.07911) (Instruction-Following Eval) is a benchmark that evaluates a model's
ability to follow verifiable, format-level instructions.  Each prompt contains one or more explicit
instructions (e.g. "write at least 300 words", "do not use commas", "respond in JSON format").

**HuggingFace dataset**: [`google/IFEval`](https://huggingface.co/datasets/google/IFEval)

## Metrics

| Metric | Description |
|---|---|
| `prompt_level_accuracy` | Fraction of prompts where **all** instructions are satisfied |
| `instruction_level_accuracy` | Fraction of individual instructions that are satisfied |
| `num_prompts` | Total number of evaluated prompts |
| `num_instructions` | Total number of evaluated instructions |

## Running the evaluation

```bash
cd evaluation
python evaluate.py \
  --dataset ifeval \
  --model meta-llama/Meta-Llama-3.1-8B-Instruct \
  --press_name snapkv \
  --compression_ratio 0.5
```

Or via `evaluate_config.yaml`:

```yaml
dataset: "ifeval"
data_dir: null
model: "meta-llama/Meta-Llama-3.1-8B-Instruct"
press_name: "snapkv"
compression_ratio: 0.5
max_new_tokens: 1280
```

## Notes

- IFEval has no long context; all prompts are short instructions.  The `context` field is set to the
  prompt itself so that every sample is processed independently.
- The evaluation uses rule-based checkers that cover all 25 instruction types defined in the original
  paper and reference implementation
  ([`google-research/instruction_following_eval`](https://github.com/google-research/google-research/tree/aa633e5105c702b47a4dd836d9b6eca39984a0fe/instruction_following_eval)).
- Language-detection instructions (`language:response_language`) require the optional `langdetect`
  package.  If it is not installed the check is skipped (returns `True`).
