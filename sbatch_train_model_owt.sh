#!/bin/bash
#SBATCH --job-name=train_lm_owt
#SBATCH --partition=fatq
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=32
#SBATCH --mem=120G
#SBATCH --time=7-00:00:00
#SBATCH --output=log/%x_%j.log
#SBATCH --error=log/%x_%j.log

set -euo pipefail
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

echo "=== Job info ==="
echo "Job ID: $SLURM_JOB_ID"
echo "Host: $(hostname)"
echo "PWD:  $(pwd)"
echo "Date: $(date)"
echo

module load cuda12.6/toolkit/12.6

echo "=== GPU info ==="
nvidia-smi
echo

PROJECT_ROOT="${SLURM_SUBMIT_DIR:-$(pwd)}"
cd "$PROJECT_ROOT"

# Conda setup
source /var/scratch/dpp2567/miniconda3/etc/profile.d/conda.sh
conda activate base

echo "=== Python / uv info ==="
which python || true
which uv || true
python --version || true
echo

TRAIN_BIN="data/owt_train_gpt2_tokens.bin"
VAL_BIN="data/owt_valid_gpt2_tokens.bin"
TRAIN_TXT="data/owt_train.txt"
VAL_TXT="data/owt_valid.txt"

echo "=== Path checks ==="
echo "PROJECT_ROOT: $PROJECT_ROOT"
echo "TRAIN_BIN:    $TRAIN_BIN"
echo "VAL_BIN:      $VAL_BIN"
echo "TRAIN_TXT:    $TRAIN_TXT"
echo "VAL_TXT:      $VAL_TXT"
echo

if [[ ! -f "$TRAIN_BIN" || ! -f "$VAL_BIN" ]]; then
  echo "GPT-2 token bin files not found. Preparing them now..."
  [[ -f "$TRAIN_TXT" ]] || { echo "Missing $TRAIN_TXT"; exit 1; }
  [[ -f "$VAL_TXT" ]] || { echo "Missing $VAL_TXT"; exit 1; }

  uv run cs336_basics/prepare_tokens_memmap_gpt2.py \
    --train_txt "$TRAIN_TXT" \
    --val_txt "$VAL_TXT" \
    --train_out "$TRAIN_BIN" \
    --val_out "$VAL_BIN" \
    --dtype uint16
fi

echo "=== Train start ==="
uv run torchrun --standalone --nproc_per_node=4 cs336_basics/train_lm.py \
  --train_data "$TRAIN_BIN" \
  --val_data "$VAL_BIN" \
  --data_dtype uint16 \
  --vocab_size 50257 \
  --context_length 256 \
  --d_model 640 \
  --num_layers 10 \
  --num_heads 10 \
  --d_ff 1920 \
  --batch_size 16 \
  --max_steps 20000 \
  --eval_every 200 \
  --eval_iters 20 \
  --log_every 20 \
  --max_lr 5e-4 \
  --min_lr 5e-5 \
  --warmup_iters 1500 \
  --cosine_cycle_iters 20000 \
  --weight_decay 0.1 \
  --amp_dtype fp16 \
  --checkpoint_path checkpoints/owt_lm_gpt2tok.pt \
  --save_every 500

echo "=== Done ==="
date
