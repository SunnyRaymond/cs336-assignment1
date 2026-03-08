from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np
import torch

from cs336_basics.model import TransformerLM, cross_entropy
from cs336_basics.optimizer import AdamW


def get_lr(it: int, max_lr: float, min_lr: float, warmup_iters: int, cosine_cycle_iters: int) -> float:
    if it < warmup_iters:
        return max_lr * it / warmup_iters
    if it <= cosine_cycle_iters:
        ratio = (it - warmup_iters) / (cosine_cycle_iters - warmup_iters)
        coeff = 0.5 * (1.0 + math.cos(math.pi * ratio))
        return min_lr + coeff * (max_lr - min_lr)
    return min_lr


def get_batch(dataset: np.memmap, batch_size: int, context_length: int, device: str) -> tuple[torch.Tensor, torch.Tensor]:
    max_start = len(dataset) - context_length - 1
    starts = np.random.randint(0, max_start + 1, size=batch_size)
    x_np = np.stack([dataset[s : s + context_length] for s in starts])
    y_np = np.stack([dataset[s + 1 : s + context_length + 1] for s in starts])
    x = torch.from_numpy(x_np.astype(np.int64, copy=False)).to(device)
    y = torch.from_numpy(y_np.astype(np.int64, copy=False)).to(device)
    return x, y


@torch.no_grad()
def estimate_loss(
    model: torch.nn.Module,
    train_data: np.memmap,
    val_data: np.memmap,
    batch_size: int,
    context_length: int,
    eval_iters: int,
    device: str,
) -> tuple[float, float]:
    model.eval()
    train_losses = []
    val_losses = []
    for _ in range(eval_iters):
        x, y = get_batch(train_data, batch_size, context_length, device)
        logits = model(x)
        train_losses.append(cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1)).item())
    for _ in range(eval_iters):
        x, y = get_batch(val_data, batch_size, context_length, device)
        logits = model(x)
        val_losses.append(cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1)).item())
    model.train()
    return float(np.mean(train_losses)), float(np.mean(val_losses))


def save_checkpoint(path: Path, model: torch.nn.Module, optimizer: torch.optim.Optimizer, step: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "step": step,
        },
        path,
    )


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Train TransformerLM with memmap datasets")
    p.add_argument("--train_data", type=Path, required=True, help="Path to train binary token file")
    p.add_argument("--val_data", type=Path, required=True, help="Path to val binary token file")
    p.add_argument("--data_dtype", type=str, default="uint16", choices=["uint16", "int32", "int64"])
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")

    p.add_argument("--vocab_size", type=int, required=True)
    p.add_argument("--context_length", type=int, default=128)
    p.add_argument("--d_model", type=int, default=256)
    p.add_argument("--num_layers", type=int, default=4)
    p.add_argument("--num_heads", type=int, default=4)
    p.add_argument("--d_ff", type=int, default=768)
    p.add_argument("--rope_theta", type=float, default=10000.0)

    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--max_steps", type=int, default=1000)
    p.add_argument("--eval_every", type=int, default=100)
    p.add_argument("--eval_iters", type=int, default=20)
    p.add_argument("--log_every", type=int, default=10)
    p.add_argument("--grad_clip", type=float, default=1.0)

    p.add_argument("--max_lr", type=float, default=3e-4)
    p.add_argument("--min_lr", type=float, default=3e-5)
    p.add_argument("--warmup_iters", type=int, default=100)
    p.add_argument("--cosine_cycle_iters", type=int, default=1000)
    p.add_argument("--weight_decay", type=float, default=0.1)
    p.add_argument("--beta1", type=float, default=0.9)
    p.add_argument("--beta2", type=float, default=0.95)
    p.add_argument("--eps", type=float, default=1e-8)

    p.add_argument("--checkpoint_path", type=Path, default=Path("checkpoints/latest.pt"))
    p.add_argument("--save_every", type=int, default=100)
    p.add_argument("--resume", action="store_true")

    p.add_argument("--use_wandb", action="store_true")
    p.add_argument("--wandb_project", type=str, default="cs336-assignment1")
    return p


def main() -> None:
    args = build_parser().parse_args()

    train_data = np.memmap(args.train_data, mode="r", dtype=args.data_dtype)
    val_data = np.memmap(args.val_data, mode="r", dtype=args.data_dtype)

    model = TransformerLM(
        vocab_size=args.vocab_size,
        context_length=args.context_length,
        d_model=args.d_model,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        d_ff=args.d_ff,
        rope_theta=args.rope_theta,
        device=torch.device(args.device),
    ).to(args.device)
    optimizer = AdamW(
        model.parameters(),
        lr=args.max_lr,
        betas=(args.beta1, args.beta2),
        eps=args.eps,
        weight_decay=args.weight_decay,
    )

    start_step = 0
    if args.resume and args.checkpoint_path.exists():
        ckpt = torch.load(args.checkpoint_path, map_location="cpu")
        model.load_state_dict(ckpt["model_state_dict"])
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        start_step = int(ckpt["step"]) + 1

    wandb_run = None
    if args.use_wandb:
        import wandb

        wandb_run = wandb.init(project=args.wandb_project, config=vars(args))

    model.train()
    for step in range(start_step, args.max_steps):
        lr = get_lr(step, args.max_lr, args.min_lr, args.warmup_iters, args.cosine_cycle_iters)
        for pg in optimizer.param_groups:
            pg["lr"] = lr

        x, y = get_batch(train_data, args.batch_size, args.context_length, args.device)
        logits = model(x)
        loss = cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1))

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        optimizer.step()

        if step % args.log_every == 0:
            print(f"step={step} lr={lr:.6g} train_loss={loss.item():.6f}")
            if wandb_run is not None:
                wandb_run.log({"step": step, "lr": lr, "train_loss": float(loss.item())}, step=step)

        if step % args.eval_every == 0:
            train_eval, val_eval = estimate_loss(
                model,
                train_data,
                val_data,
                args.batch_size,
                args.context_length,
                args.eval_iters,
                args.device,
            )
            print(f"[eval] step={step} train={train_eval:.6f} val={val_eval:.6f}")
            if wandb_run is not None:
                wandb_run.log({"step": step, "eval_train_loss": train_eval, "eval_val_loss": val_eval}, step=step)

        if step % args.save_every == 0:
            save_checkpoint(args.checkpoint_path, model, optimizer, step)

    save_checkpoint(args.checkpoint_path, model, optimizer, args.max_steps - 1)
    if wandb_run is not None:
        wandb_run.finish()


if __name__ == "__main__":
    main()
