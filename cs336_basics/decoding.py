from __future__ import annotations

import torch

from cs336_basics.bpe import Tokenizer


def _sample_top_p(probs: torch.Tensor, top_p: float) -> torch.Tensor:
    if not (0.0 < top_p <= 1.0):
        raise ValueError("top_p must be in (0, 1].")

    sorted_probs, sorted_idx = torch.sort(probs, dim=-1, descending=True)
    cdf = torch.cumsum(sorted_probs, dim=-1)
    keep = cdf <= top_p
    keep[..., 0] = True

    filtered = torch.where(keep, sorted_probs, torch.zeros_like(sorted_probs))
    filtered = filtered / filtered.sum(dim=-1, keepdim=True)
    sampled_sorted = torch.multinomial(filtered, num_samples=1)
    return sorted_idx.gather(-1, sampled_sorted).squeeze(-1)


@torch.no_grad()
def generate_ids(
    model: torch.nn.Module,
    prompt_ids: list[int],
    max_new_tokens: int,
    temperature: float = 1.0,
    top_p: float = 1.0,
    eos_token_id: int | None = None,
    device: str | torch.device | None = None,
) -> list[int]:
    if max_new_tokens < 0:
        raise ValueError("max_new_tokens must be >= 0.")
    if temperature < 0:
        raise ValueError("temperature must be >= 0.")

    if device is None:
        device = next(model.parameters()).device
    model.eval()

    tokens = torch.tensor([prompt_ids], dtype=torch.long, device=device)
    for _ in range(max_new_tokens):
        model_in = tokens
        if hasattr(model, "context_length") and tokens.shape[1] > model.context_length:
            model_in = tokens[:, -model.context_length :]
        logits = model(model_in)
        next_logits = logits[:, -1, :]

        if temperature == 0:
            next_token = torch.argmax(next_logits, dim=-1)
        else:
            next_logits = next_logits / temperature
            probs = torch.softmax(next_logits, dim=-1)
            if top_p < 1.0:
                next_token = _sample_top_p(probs, top_p).view(1)
            else:
                next_token = torch.multinomial(probs, num_samples=1).squeeze(-1)

        tokens = torch.cat([tokens, next_token.view(1, 1)], dim=1)
        if eos_token_id is not None and next_token.item() == eos_token_id:
            break

    return tokens[0].tolist()


@torch.no_grad()
def generate_text(
    model: torch.nn.Module,
    tokenizer: Tokenizer,
    prompt: str,
    max_new_tokens: int,
    temperature: float = 1.0,
    top_p: float = 1.0,
    eos_token: str = "<|endoftext|>",
    device: str | torch.device | None = None,
) -> str:
    prompt_ids = tokenizer.encode(prompt)
    eos_token_id = tokenizer.special_to_id.get(eos_token)
    out_ids = generate_ids(
        model=model,
        prompt_ids=prompt_ids,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_p=top_p,
        eos_token_id=eos_token_id,
        device=device,
    )
    return tokenizer.decode(out_ids)
