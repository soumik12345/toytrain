"""Shared utilities used by the dataset-local preparation scripts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from datasets import (
    Dataset,
    DatasetDict,
    concatenate_datasets,
    get_dataset_config_names,
    load_dataset,
)
from transformers import AutoTokenizer

SMOKE_TEST_SAMPLES = 100


def count_tokens(dataset: Dataset) -> int:
    return sum(
        sum(len(input_ids) for input_ids in batch["input_ids"])
        for batch in dataset.iter(batch_size=1_000)
    )


def build_parser(dataset_id: str, output_dir: Path) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=f"Tokenize {dataset_id} with Qwen.")
    parser.add_argument("--tokenizer", default="Qwen/Qwen3-0.6B")
    parser.add_argument("--output-dir", type=Path, default=output_dir)
    parser.add_argument("--num-proc", type=int, default=8)
    parser.add_argument("--validation-fraction", type=float, default=0.0005)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--subsets", nargs="+", help="Configurations to process; defaults to all.")
    parser.add_argument(
        "--smoke-test",
        action="store_true",
        help=f"Load only the first {SMOKE_TEST_SAMPLES} documents from each subset.",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        help="Load at most this many documents from each subset.",
    )
    return parser


def prepare_dataset(dataset_id: str, args: argparse.Namespace) -> None:

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, use_fast=True)
    configs = args.subsets or get_dataset_config_names(dataset_id)
    max_samples = SMOKE_TEST_SAMPLES if args.smoke_test else args.max_samples
    if max_samples is not None and max_samples < 2:
        raise ValueError("--max-samples must be at least 2")
    tokenized_configs = []
    skipped = []
    for config in configs:
        split_name = "train" if max_samples is None else f"train[:{max_samples}]"
        dataset = load_dataset(dataset_id, config, split=split_name)
        text_column = next((name for name in ("text", "content") if name in dataset.column_names), None)
        if text_column is None:
            skipped.append({"subset": config, "columns": dataset.column_names})
            continue
        split = dataset.train_test_split(
            test_size=args.validation_fraction, seed=args.seed, shuffle=True
        )
        split["validation"] = split.pop("test")
        source = f"{dataset_id}/{config}"

        def tokenize_batch(batch: dict[str, list[str]]) -> dict[str, list]:
            encoded = tokenizer(
                batch[text_column],
                add_special_tokens=False,
                truncation=False,
                return_attention_mask=False,
                return_token_type_ids=False,
            )
            return {
                "input_ids": [ids + [tokenizer.eos_token_id] for ids in encoded["input_ids"]],
                "source": [source] * len(batch[text_column]),
            }

        tokenized_configs.append(
            split.map(
                tokenize_batch,
                batched=True,
                batch_size=256,
                remove_columns=dataset.column_names,
                desc=f"Tokenizing {source}",
                num_proc=args.num_proc,
            )
        )
    if not tokenized_configs:
        raise ValueError(f"No text/content subsets found for {dataset_id}; skipped: {skipped}")

    tokenized = DatasetDict(
        {
            split: concatenate_datasets([dataset[split] for dataset in tokenized_configs])
            for split in ("train", "validation")
        }
    )
    tokenized.save_to_disk(args.output_dir)
    tokenizer.save_pretrained(args.output_dir / "tokenizer")
    statistics = {
        split: {"documents": dataset.num_rows, "tokens": count_tokens(dataset)}
        for split, dataset in tokenized.items()
    }
    (args.output_dir / "metadata.json").write_text(
        json.dumps(
            {
                "dataset": dataset_id,
                "subsets": configs,
                "skipped_subsets": skipped,
                "tokenizer": tokenizer.name_or_path,
                "vocab_size": len(tokenizer),
                "eos_token_id": tokenizer.eos_token_id,
                "splits": statistics,
                "max_samples_per_subset": max_samples,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    for split, values in statistics.items():
        print(f"{split}: {values['documents']:,} documents, {values['tokens']:,} tokens")
