from __future__ import annotations

import argparse
from pathlib import Path

import torch

from cs336_basics.bpe import Tokenizer
from cs336_basics.decoding import generate_ids
from cs336_basics.model import TransformerLM


def _resolve_device(device: str) -> str:
    if device == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return device


def _strip_ddp_prefix(state_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    # Train script may save from DDP-wrapped model with "module." key prefixes.
    if not state_dict:
        return state_dict
    if next(iter(state_dict)).startswith("module."):
        return {k.removeprefix("module."): v for k, v in state_dict.items()}
    return state_dict


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Generate text from a saved TransformerLM checkpoint.")
    p.add_argument("--checkpoint_path", type=Path, required=True)
    p.add_argument("--tokenizer_pkl", type=Path, required=True)

    p.add_argument("--prompt", type=str, default="Once upon a time")
    p.add_argument("--max_new_tokens", type=int, default=256)
    p.add_argument("--temperature", type=float, default=0.8)
    p.add_argument("--top_p", type=float, default=0.95)
    p.add_argument("--eos_token", type=str, default="<|endoftext|>")
    p.add_argument("--seed", type=int, default=1337)
    p.add_argument("--device", type=str, default="auto")
    p.add_argument("--out_file", type=Path, default=None)

    # Model architecture (defaults match sbatch_train_model.sh in this repo).
    p.add_argument("--vocab_size", type=int, default=10000)
    p.add_argument("--context_length", type=int, default=256)
    p.add_argument("--d_model", type=int, default=640)
    p.add_argument("--num_layers", type=int, default=10)
    p.add_argument("--num_heads", type=int, default=10)
    p.add_argument("--d_ff", type=int, default=1920)
    p.add_argument("--rope_theta", type=float, default=10000.0)
    return p


def main() -> None:
    args = _build_parser().parse_args()
    torch.manual_seed(args.seed)
    device = _resolve_device(args.device)

    tokenizer = Tokenizer.from_files(
        vocab_filepath=str(args.tokenizer_pkl),
        merges_filepath=str(args.tokenizer_pkl),
        special_tokens=[args.eos_token],
    )

    model = TransformerLM(
        vocab_size=args.vocab_size,
        context_length=args.context_length,
        d_model=args.d_model,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        d_ff=args.d_ff,
        rope_theta=args.rope_theta,
        device=torch.device(device),
    ).to(device)

    ckpt = torch.load(args.checkpoint_path, map_location="cpu")
    state_dict = ckpt["model_state_dict"] if isinstance(ckpt, dict) and "model_state_dict" in ckpt else ckpt
    state_dict = _strip_ddp_prefix(state_dict)
    model.load_state_dict(state_dict, strict=True)
    model.eval()

    prompt_ids = tokenizer.encode(args.prompt)
    if not prompt_ids:
        eos_id = tokenizer.special_to_id.get(args.eos_token)
        if eos_id is None:
            raise ValueError("Prompt is empty and eos_token is missing in tokenizer.")
        prompt_ids = [eos_id]

    eos_id = tokenizer.special_to_id.get(args.eos_token)
    out_ids = generate_ids(
        model=model,
        prompt_ids=prompt_ids,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        eos_token_id=eos_id,
        device=device,
    )
    generated_text = tokenizer.decode(out_ids)
    new_tokens = len(out_ids) - len(prompt_ids)

    header = (
        f"# prompt_tokens={len(prompt_ids)} new_tokens={new_tokens} total_tokens={len(out_ids)} "
        f"temperature={args.temperature} top_p={args.top_p}\n"
    )
    output = header + generated_text + "\n"

    if args.out_file is not None:
        args.out_file.parent.mkdir(parents=True, exist_ok=True)
        args.out_file.write_text(output, encoding="utf-8")
        print(f"Wrote generation to: {args.out_file}")
    else:
        print(output)


if __name__ == "__main__":
    main()
