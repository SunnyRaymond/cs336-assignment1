#!/bin/bash
#SBATCH --job-name=gen_lm_owt
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=24G
#SBATCH --time=7-00:00:00
#SBATCH --output=log/%x_%j.log
#SBATCH --error=log/%x_%j.log

set -euo pipefail
export PYTHONUNBUFFERED=1

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

source /var/scratch/dpp2567/miniconda3/etc/profile.d/conda.sh
conda activate base

echo "=== Python / uv info ==="
which python || true
which uv || true
python --version || true
echo

CHECKPOINT_PATH="checkpoints/owt_lm_gpt2tok.pt"
TOKENIZER_VOCAB="tests/fixtures/gpt2_vocab.json"
TOKENIZER_MERGES="tests/fixtures/gpt2_merges.txt"
PROMPT="In a shocking turn of events,"
MAX_NEW_TOKENS=256
TEMPERATURE=0.8
TOP_P=0.95
OUT_FILE="log/generate_owt_${SLURM_JOB_ID}.txt"

echo "=== Path checks ==="
echo "PROJECT_ROOT:      $PROJECT_ROOT"
echo "CHECKPOINT_PATH:   $CHECKPOINT_PATH"
echo "TOKENIZER_VOCAB:   $TOKENIZER_VOCAB"
echo "TOKENIZER_MERGES:  $TOKENIZER_MERGES"
echo "OUT_FILE:          $OUT_FILE"
echo

[[ -f "$CHECKPOINT_PATH" ]] || { echo "Missing $CHECKPOINT_PATH"; exit 1; }
[[ -f "$TOKENIZER_VOCAB" ]] || { echo "Missing $TOKENIZER_VOCAB"; exit 1; }
[[ -f "$TOKENIZER_MERGES" ]] || { echo "Missing $TOKENIZER_MERGES"; exit 1; }

echo "=== Generate start ==="
uv run cs336_basics/generate_lm.py \
  --checkpoint_path "$CHECKPOINT_PATH" \
  --tokenizer_vocab_path "$TOKENIZER_VOCAB" \
  --tokenizer_merges_path "$TOKENIZER_MERGES" \
  --prompt "$PROMPT" \
  --max_new_tokens "$MAX_NEW_TOKENS" \
  --temperature "$TEMPERATURE" \
  --top_p "$TOP_P" \
  --vocab_size 50257 \
  --context_length 256 \
  --d_model 640 \
  --num_layers 10 \
  --num_heads 10 \
  --d_ff 1920 \
  --out_file "$OUT_FILE"

echo "=== Generated sample ==="
head -n 40 "$OUT_FILE" || true
echo

echo "=== Done ==="
date

