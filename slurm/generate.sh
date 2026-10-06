#!/bin/bash
#SBATCH --job-name=ddpm_hi_generate
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --time=01:00:00
#SBATCH --output=slurm-generate-%j.out
#SBATCH --error=slurm-generate-%j.err
# Add --account / --partition lines for your cluster.
#
# Runs generate_map_pk.py with whatever arguments are passed. Submit from the repository root, e.g.
#   sbatch slurm/generate.sh --checkpoint weights/2param/model.pt --labels 0.3 0.8 --n_maps 16

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")/..}"
"${PYTHON:-python}" -u generate_map_pk.py "$@"
