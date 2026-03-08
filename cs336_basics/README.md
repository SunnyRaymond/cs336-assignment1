# Training Script Notes

This folder includes [`train_lm.py`](./train_lm.py), a minimal training loop for `TransformerLM`.

## Memmap Data Format

`train_lm.py` expects tokenized corpora stored as a **flat 1D binary array** readable by `numpy.memmap`:

- one integer token id per element
- no headers/metadata in the file
- dtype must match `--data_dtype` (`uint16`, `int32`, or `int64`)

Example:

```python
import numpy as np

tokens = np.array([12, 55, 9, 402, 17], dtype=np.uint16)
tokens.tofile("train_tokens.bin")
```

Then launch:

```powershell
uv run cs336_basics/train_lm.py `
  --train_data train_tokens.bin `
  --val_data val_tokens.bin `
  --data_dtype uint16 `
  --vocab_size 10000
```

## Checkpointing

- save path is controlled by `--checkpoint_path`
- periodic saves are controlled by `--save_every`
- resume with `--resume`

Example:

```powershell
uv run cs336_basics/train_lm.py `
  --train_data train_tokens.bin `
  --val_data val_tokens.bin `
  --data_dtype uint16 `
  --vocab_size 10000 `
  --checkpoint_path checkpoints/latest.pt `
  --resume
```

## Decoding

Use [`decoding.py`](./decoding.py) for generation with:

- prompt continuation
- `max_new_tokens`
- temperature scaling
- top-p (nucleus) sampling
- optional stop on `<|endoftext|>`
