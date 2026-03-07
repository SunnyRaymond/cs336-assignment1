import argparse
import pickle
from pathlib import Path

from tests.adapters import run_train_bpe


def train_one(input_path: str, output_path: str, vocab_size: int, special_tokens: list[str]) -> None:
    vocab, merges = run_train_bpe(
        input_path=input_path,
        vocab_size=vocab_size,
        special_tokens=special_tokens,
    )

    out_path = Path(output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("wb") as f:
        pickle.dump({"vocab": vocab, "merges": merges}, f)

    print(f"Saved {out_path} | vocab_size={len(vocab)} | num_merges={len(merges)}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Train BPE on TinyStories and/or OpenWebText.")

    parser.add_argument(
        "--dataset",
        choices=["tinystories", "owt", "both"],
        default="tinystories",
        help="Which dataset to train on."
    )

    parser.add_argument("--vocab-size", type=int, default=10000)

    parser.add_argument(
        "--special-token",
        action="append",
        default=["<|endoftext|>"],
        help="Special token to include. Can be provided multiple times.",
    )

    parser.add_argument("--tinystories-input", default="data/TinyStoriesV2-GPT4-train.txt")
    parser.add_argument("--owt-input", default="data/owt_train.txt")

    parser.add_argument("--tinystories-output", default="tinystories_bpe.pkl")
    parser.add_argument("--owt-output", default="owt_bpe.pkl")

    args = parser.parse_args()

    if args.dataset in ["tinystories", "both"]:
        train_one(
            input_path=args.tinystories_input,
            output_path=args.tinystories_output,
            vocab_size=args.vocab_size,
            special_tokens=args.special_token,
        )

    if args.dataset in ["owt", "both"]:
        train_one(
            input_path=args.owt_input,
            output_path=args.owt_output,
            vocab_size=args.vocab_size,
            special_tokens=args.special_token,
        )


if __name__ == "__main__":
    main()