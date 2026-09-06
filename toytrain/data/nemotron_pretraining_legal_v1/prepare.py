from pathlib import Path

from toytrain.data.utils import build_parser, prepare_dataset

DATASET_ID = "nvidia/Nemotron-Pretraining-Legal-v1"


def main() -> None:
    args = build_parser(DATASET_ID, Path(__file__).with_name("qwen_tokenized")).parse_args()
    prepare_dataset(DATASET_ID, args)


if __name__ == "__main__":
    main()
