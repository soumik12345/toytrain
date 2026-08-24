# Nemotron pretraining sample

[Dataset card](https://huggingface.co/datasets/nvidia/Nemotron-Pretraining-Dataset-sample)

Tokenize all configured pretraining subsets with the default Qwen3 tokenizer:

```bash
uv run python toytrain/data/nemotron_pretraining_data_sample/prepare.py
```

Use a different Qwen tokenizer or output directory for an experiment:

```bash
uv run python toytrain/data/nemotron_pretraining_data_sample/prepare.py \
  --tokenizer Qwen/Qwen2.5-0.5B \
  --output-dir data/qwen25_experiment
```

The output contains `train` and `validation` splits with `input_ids` and
`source` columns, a local tokenizer snapshot, and `metadata.json`. Configure the
transformer's vocabulary size and special-token IDs from that metadata. Pack or
crop `input_ids` into the desired context length in the training input pipeline.
