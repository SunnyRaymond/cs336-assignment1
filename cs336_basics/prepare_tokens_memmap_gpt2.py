from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import tiktoken


def encode_text_to_bin_gpt2(
    encoding: tiktoken.Encoding,
    input_path: Path,
    output_path: Path,
    dtype: np.dtype,
    flush_tokens: int,
) -> int:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    total = 0
    buf: list[int] = []

    with input_path.open("r", encoding="utf-8", errors="replace") as fin, output_path.open("wb") as fout:
        for line in fin:
            ids = encoding.encode(line, allowed_special={"<|endoftext|>"})
            if not ids:
                continue
            buf.extend(ids)
            if len(buf) >= flush_tokens:
                arr = np.asarray(buf, dtype=dtype)
                arr.tofile(fout)
                total += int(arr.size)
                buf.clear()

        if buf:
            arr = np.asarray(buf, dtype=dtype)
            arr.tofile(fout)
            total += int(arr.size)

    return total


def main() -> None:
    p = argparse.ArgumentParser(description="Tokenize text files with GPT-2 tokenizer into flat binary token-id arrays.")
    p.add_argument("--train_txt", type=Path, required=True)
    p.add_argument("--val_txt", type=Path, required=True)
    p.add_argument("--train_out", type=Path, default=Path("data/owt_train_gpt2_tokens.bin"))
    p.add_argument("--val_out", type=Path, default=Path("data/owt_valid_gpt2_tokens.bin"))
    p.add_argument("--dtype", choices=["uint16", "int32", "int64"], default="uint16")
    p.add_argument("--flush_tokens", type=int, default=1_000_000)
    args = p.parse_args()

    encoding = tiktoken.get_encoding("gpt2")
    dtype = np.dtype(args.dtype)

    n_train = encode_text_to_bin_gpt2(encoding, args.train_txt, args.train_out, dtype, args.flush_tokens)
    n_val = encode_text_to_bin_gpt2(encoding, args.val_txt, args.val_out, dtype, args.flush_tokens)

    print(f"tokenizer=gpt2 vocab_size={encoding.n_vocab}")
    print(f"train_out={args.train_out} tokens={n_train} dtype={args.dtype}")
    print(f"val_out={args.val_out} tokens={n_val} dtype={args.dtype}")


if __name__ == "__main__":
    main()
