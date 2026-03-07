#!/bin/bash
#SBATCH --job-name=train_bpe_both
#SBATCH --partition=defq
#SBATCH --constraint=TitanRTX
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=100G
#SBATCH --time=12:00:00
#SBATCH --output=log/%x_%j.out
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

source /var/scratch/dpp2567/miniconda3/etc/profile.d/conda.sh

# Go to your project root if needed
# Example:
# cd /var/scratch/dpp2567/CS336-2025-Spring/assignment1-basics

# Conda setup
source /var/scratch/dpp2567/miniconda3/etc/profile.d/conda.sh

# Activate your environment if needed
# conda activate cs336-lec

echo "=== Python / uv info ==="
which python || true
which uv || true
python --version || true
echo

echo "=== Train TinyStories BPE (vocab_size=10000) ==="
uv run python train_bpe.py \
  --dataset tinystories \
  --vocab-size 10000 \
  --tinystories-output tinystories_bpe.pkl
echo

echo "=== Train OpenWebText BPE (vocab_size=32000) ==="
uv run python train_bpe.py \
  --dataset owt \
  --vocab-size 32000 \
  --owt-output owt_bpe.pkl
echo

echo "=== Done ==="
date