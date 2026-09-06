"""Tokenize the Nemotron pretraining sample with a Qwen tokenizer."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from datasets import Dataset, DatasetDict, concatenate_datasets, load_dataset
from transformers import AutoTokenizer, PreTrainedTokenizerBase

DATASET_ID = "nvidia/Nemotron-Pretraining-Dataset-sample"
DEFAULT_TOKENIZER = "Qwen/Qwen3-0.6B"
SMOKE_TEST_SAMPLES = 100
DEFAULT_SUBSETS = (
    "Nemotron-CC-MATH",
    "Nemotron-CC-High-Quality",
    "Nemotron-CC-High-Quality-Synthetic",
    "Nemotron-CC-Diverse-QA",
    "Nemotron-CC-Translated-Diverse-QA",
    "Nemotron-Synthetic-Code",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tokenizer", default=DEFAULT_TOKENIZER)
    parser.add_argument(
        "--output-dir", type=Path, default=Path(__file__).with_name("qwen_tokenized")
    )
    parser.add_argument("--num-proc", type=int, default=8)
    parser.add_argument("--validation-fraction", type=float, default=0.0005)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--subsets", nargs="+", default=list(DEFAULT_SUBSETS))
    parser.add_argument("--smoke-test", action="store_true")
    parser.add_argument("--max-samples", type=int)
    return parser.parse_args()


def tokenize_subset(
    subset: str,
    tokenizer: PreTrainedTokenizerBase,
    *,
    validation_fraction: float,
    seed: int,
    num_proc: int,
    max_samples: int | None,
) -> DatasetDict:
    split_name = "train" if max_samples is None else f"train[:{max_samples}]"
    dataset = load_dataset(DATASET_ID, subset, split=split_name)
    if "text" not in dataset.column_names:
        raise ValueError(f"{subset!r} has no 'text' column; found {dataset.column_names}")
    split = dataset.train_test_split(
        test_size=validation_fraction, seed=seed, shuffle=True
    )
    split["validation"] = split.pop("test")
    eos_token_id = tokenizer.eos_token_id
    if eos_token_id is None:
        raise ValueError(f"Tokenizer {tokenizer.name_or_path!r} has no EOS token")

    def tokenize_batch(batch: dict[str, list[str]]) -> dict[str, list]:
        encoded = tokenizer(
            batch["text"],
            add_special_tokens=False,
            truncation=False,
            return_attention_mask=False,
            return_token_type_ids=False,
        )
        return {
            "input_ids": [ids + [eos_token_id] for ids in encoded["input_ids"]],
            "source": [subset] * len(batch["text"]),
        }

    return split.map(
        tokenize_batch,
        batched=True,
        batch_size=256,
        remove_columns=dataset.column_names,
        desc=f"Tokenizing {subset}",
        num_proc=num_proc,
    )


def count_tokens(dataset: Dataset) -> int:
    return sum(
        sum(len(input_ids) for input_ids in batch["input_ids"])
        for batch in dataset.iter(batch_size=1_000)
    )


def main() -> None:
    args = parse_args()
    max_samples = SMOKE_TEST_SAMPLES if args.smoke_test else args.max_samples
    if max_samples is not None and max_samples < 2:
        raise ValueError("--max-samples must be at least 2")
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, use_fast=True)
    tokenized_subsets = [
        tokenize_subset(
            subset,
            tokenizer,
            validation_fraction=args.validation_fraction,
            seed=args.seed,
            num_proc=args.num_proc,
            max_samples=max_samples,
        )
        for subset in args.subsets
    ]
    tokenized = DatasetDict(
        {
            split: concatenate_datasets([dataset[split] for dataset in tokenized_subsets])
            for split in ("train", "validation")
        }
    )
    tokenized.save_to_disk(args.output_dir)
    tokenizer.save_pretrained(args.output_dir / "tokenizer")
    metadata = {
        "dataset": DATASET_ID,
        "subsets": args.subsets,
        "tokenizer": tokenizer.name_or_path,
        "vocab_size": len(tokenizer),
        "bos_token_id": tokenizer.bos_token_id,
        "eos_token_id": tokenizer.eos_token_id,
        "pad_token_id": tokenizer.pad_token_id,
        "max_samples_per_subset": max_samples,
    }
    (args.output_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
    )
    print(tokenized)
    for split, dataset in tokenized.items():
        print(f"{split}: {dataset.num_rows:,} documents, {count_tokens(dataset):,} tokens")


if __name__ == "__main__":
    main()
