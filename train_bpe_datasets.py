import argparse
import multiprocessing as mp
import os
import pickle
import time
from pathlib import Path

from tests.adapters import run_train_bpe


def _human_bytes(num_bytes: int) -> str:
    units = ["B", "KB", "MB", "GB", "TB"]
    value = float(num_bytes)
    for unit in units:
        if value < 1024.0 or unit == units[-1]:
            return f"{value:.2f} {unit}"
        value /= 1024.0
    return f"{num_bytes} B"


def train_one(
    dataset_name: str,
    input_path: str,
    output_path: str,
    vocab_size: int,
    special_tokens: list[str],
    workers: int,
    device: str,
) -> None:
    input_file = Path(input_path)
    input_size = input_file.stat().st_size if input_file.exists() else 0
    target_merges = max(0, vocab_size - 256 - len(special_tokens))
    start = time.perf_counter()

    print(f"[{dataset_name}] Training start")
    print(f"[{dataset_name}] Input: {input_file}")
    print(f"[{dataset_name}] Input size: {_human_bytes(input_size)}")
    print(
        f"[{dataset_name}] vocab_size={vocab_size} special_tokens={special_tokens} "
        f"target_merges={target_merges} workers={workers} device={device}"
    )

    vocab, merges = run_train_bpe(
        input_path=input_path,
        vocab_size=vocab_size,
        special_tokens=special_tokens,
        workers=workers,
        device=device,
    )
    elapsed = time.perf_counter() - start

    out_path = Path(output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("wb") as f:
        pickle.dump({"vocab": vocab, "merges": merges}, f)
    output_size = out_path.stat().st_size if out_path.exists() else 0

    print(f"[{dataset_name}] Training done in {elapsed:.2f}s")
    print(
        f"[{dataset_name}] Saved {out_path} | artifact_size={_human_bytes(output_size)} "
        f"| learned_vocab={len(vocab)} | learned_merges={len(merges)}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Train BPE on TinyStories and/or OpenWebText.")

    parser.add_argument(
        "--dataset",
        choices=["tinystories", "owt", "both"],
        default="tinystories",
        help="Which dataset to train on."
    )

    parser.add_argument("--vocab-size", type=int, default=10000)
    parser.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    parser.add_argument(
        "--device",
        default="auto",
        help='BPE device: "auto", "cpu", "cuda", "cuda:0", "off".',
    )
    parser.add_argument(
        "--parallel-datasets",
        action="store_true",
        help="If --dataset both, train TinyStories and OWT in parallel processes.",
    )

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

    jobs = []
    if args.dataset in ["tinystories", "both"]:
        jobs.append(
            {
                "dataset_name": "TinyStories",
                "input_path": args.tinystories_input,
                "output_path": args.tinystories_output,
            }
        )
    if args.dataset in ["owt", "both"]:
        jobs.append(
            {
                "dataset_name": "OpenWebText",
                "input_path": args.owt_input,
                "output_path": args.owt_output,
            }
        )

    if args.parallel_datasets and len(jobs) > 1:
        ctx = mp.get_context("spawn")
        procs = []
        for job in jobs:
            p = ctx.Process(
                target=train_one,
                kwargs={
                    **job,
                    "vocab_size": args.vocab_size,
                    "special_tokens": args.special_token,
                    "workers": args.workers,
                    "device": args.device,
                },
            )
            p.start()
            procs.append(p)
        for p in procs:
            p.join()
        failed = [p.pid for p in procs if p.exitcode != 0]
        if failed:
            raise RuntimeError(f"Dataset training failed in process(es): {failed}")
    else:
        for job in jobs:
            train_one(
                dataset_name=job["dataset_name"],
                input_path=job["input_path"],
                output_path=job["output_path"],
                vocab_size=args.vocab_size,
                special_tokens=args.special_token,
                workers=args.workers,
                device=args.device,
            )


if __name__ == "__main__":
    main()
