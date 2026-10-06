#!/bin/bash
#SBATCH --job-name=ddpm_hi_figures
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --time=12:00:00
#SBATCH --output=slurm-figures-%j.out
#SBATCH --error=slurm-figures-%j.err
# Add --account / --partition lines for your cluster.
#
# Regenerates every Chapter 4 figure into $OUT (default results/), in the order of the thesis.
# Expects the CAMELS splits in data/ (prepare_data.py) and the released weights in weights/.
# About 6.5 GPU-hours on one NVIDIA L40S; each figure caches its maps or spectra, so after a crash
# the finished figures can be redrawn with their --from_cache / --plot_only option.
# Submit from the repository root:  sbatch slurm/figures.sh
# Optional: CAMELS_1P_MAPS / CAMELS_1P_PARAMS for the CAMELS envelopes in Fig. 4.11.

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")/..}"
PY="${PYTHON:-python}"
OUT="${OUT:-results}"
step() { echo; echo "=== $1  ($(date +%H:%M:%S))"; }

step "Fig 4.2 training curves"; "$PY" -u figures/fig4_2_training_curves.py --out "$OUT/fig4_2/training_val_overlay.png"
step "Fig 4.3 sampler study";   "$PY" -u figures/fig4_3_sampler_study.py --out_dir "$OUT/fig4_3"
step "Fig 4.4/4.5 map grids";   "$PY" -u figures/fig4_4_4_5_map_grids.py --out_dir "$OUT/fig4_4_4_5"
step "Fig 4.6 power spectrum";  "$PY" -u figures/fig4_6_power_spectrum.py --out_dir "$OUT/fig4_6" --n_gen 300
step "Fig 4.8 fidelity map";    for m in ddpm2 ddpm6; do "$PY" -u figures/fig4_8_fidelity_map.py --model "$m" --out_dir "$OUT/fig4_8"; done
step "Fig 4.10 pixel PDF";      "$PY" -u figures/fig4_10_pixel_pdf.py --out_dir "$OUT/fig4_10" --n_gen 300 --display_smooth 1.5
CAMELS_ARGS=()
if [[ -n "${CAMELS_1P_MAPS:-}" && -n "${CAMELS_1P_PARAMS:-}" ]]; then
    CAMELS_ARGS=(--camels_1p_maps "$CAMELS_1P_MAPS" --camels_1p_params "$CAMELS_1P_PARAMS")
fi
step "Fig 4.11 sensitivity";    "$PY" -u figures/fig4_11_sensitivity.py --out_dir "$OUT/fig4_11" ${CAMELS_ARGS[@]+"${CAMELS_ARGS[@]}"}

echo; echo "done: $(date)"
