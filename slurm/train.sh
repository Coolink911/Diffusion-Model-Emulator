#!/bin/bash
#SBATCH --job-name=ddpm_hi_train
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --time=48:00:00
#SBATCH --output=slurm-%j.out
#SBATCH --error=slurm-%j.err
# Add --account / --partition lines for your cluster. The thesis runs used one NVIDIA L40S (48 GB).
#
# Runs train_conditional.py with whatever arguments are passed; README.md lists the exact
# arguments for each thesis run. Submit from the repository root, e.g.
#   sbatch slurm/train.sh --label_dim 2 --data_dir data/params_2 --epochs 200 ...
# Set PYTHON to choose the interpreter (default: python on PATH).

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")/..}"

echo "job ${SLURM_JOB_ID:-interactive} on $(hostname), started $(date)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || true

"${PYTHON:-python}" -u train_conditional.py "$@"

echo "finished $(date)"
