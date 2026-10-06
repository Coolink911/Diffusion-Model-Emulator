# Conditional DDPMs for CAMELS HI maps (DDPM-2 and DDPM-6)

Code to rebuild the training data, train both diffusion models and generate maps with their
power spectra, as used in the thesis.

| Model | Conditioned on | Thesis checkpoint |
|---|---|---|
| DDPM-2 | Ωₘ, σ₈ | `checkpoint_epoch_200.pt` of the resumed run (EMA weights) |
| DDPM-6 | Ωₘ, σ₈, A_SN1, A_AGN1, A_SN2, A_AGN2 | `best_model.pt` (epoch 170, EMA weights) |

The trained weights are published at <https://huggingface.co/collins909/DDPM-HI-models>, so
step 4 below can be run without retraining.

## Files

| File | Purpose |
|---|---|
| `prepare_data.py` | raw CAMELS maps and parameters → normalised train/val/test splits |
| `dataset_conditional.py` | loads the splits; images to [-1, 1], labels z-scored with train statistics |
| `unet_conditional.py` | conditional U-Net (timestep + label embedding, attention at the two lowest resolutions) |
| `diffusion_conditional.py` | linear-β diffusion with 1500 steps; DDPM loss, ancestral and DDIM samplers |
| `train_conditional.py` | training loop (AdamW, cosine LR, EMA, gradient clipping, optional AMP, resume) |
| `generate_map_pk.py` | sample maps at any parameters, compute P(k), save arrays and a figure |
| `slurm/train.sh`, `slurm/generate.sh` | Slurm wrappers that pass their arguments through |
| `figures/` | one script per results-chapter figure, plus `common.py` and the loss-curve data (section 5) |
| `slurm/figures.sh` | regenerates every results-chapter figure in one job |
| `thesis_figures/` | the figures and tables as they appear in the thesis, with an index |

## 1. Environment

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

Training needs a CUDA GPU. The thesis runs used one NVIDIA L40S at about 16–18 min per epoch,
so 200 epochs take roughly 55–60 GPU-hours.

## 2. Data

Download two files from the CAMELS Multifield Dataset
(<https://camels-multifield-dataset.readthedocs.io>), IllustrisTNG suite, LH set:

- `Maps_HI_IllustrisTNG_LH_z=0.00.npy`: 15 000 maps, 256 × 256, 15 per simulation
- `params_LH_IllustrisTNG.txt`: 1000 simulations × 6 parameters

```bash
python prepare_data.py --maps Maps_HI_IllustrisTNG_LH_z=0.00.npy \
                       --params params_LH_IllustrisTNG.txt --out_dir data
```

This writes `data/params_2/` and `data/params_6/` (13 500 / 750 / 750 maps). The split is
seeded (`default_rng(42)`), so it is the same split the models were trained on. Each pixel
is `log10(N_HI + 1e-8)`, min-max normalised over all 15 000 maps **at each pixel position
separately**. The docstring of `prepare_data.py` gives the details.

## 3. Training

Every run writes `<output_dir>_<timestamp>/` containing `args.json`, `checkpoints/`
(`checkpoint_latest.pt`, `best_model.pt`, and `checkpoint_epoch_<N>.pt` every 20 epochs),
`samples/` and `losses.png`.

### DDPM-2 (two jobs, because of the 48 h job limit)

Job 1 was planned as 200 epochs and reached the 48 h limit after epoch 160:

```bash
sbatch slurm/train.sh --label_dim 2 --data_dir data/params_2 \
    --epochs 200 --batch_size 8 --lr 2e-4 --timesteps 1500 \
    --early_stop_patience 100 --sample_every 100 --output_dir outputs_ddpm2
```

Job 2 resumed from the epoch-160 snapshot. It extended the run to 210 epochs and rebuilt the
cosine LR schedule for that length:

```bash
sbatch slurm/train.sh --label_dim 2 --data_dir data/params_2 \
    --epochs 210 --batch_size 8 --lr 2e-4 --timesteps 1500 \
    --early_stop_patience 100 --sample_every 10 --output_dir outputs_ddpm2 \
    --resume outputs_ddpm2_<job-1 timestamp>/checkpoints/checkpoint_epoch_160.pt \
    --resume_refresh_scheduler
```

The thesis model is `checkpoints/checkpoint_epoch_200.pt` from job 2. The learning rate
follows a 200-epoch cosine schedule up to epoch 160 and a 210-epoch one after that. An
uninterrupted 200-epoch run would therefore not use the same learning rates.

### DDPM-6 (one job)

```bash
sbatch slurm/train.sh --label_dim 6 --data_dir data/params_6 \
    --epochs 200 --batch_size 8 --lr 2e-4 --timesteps 1500 \
    --early_stop_patience 100 --sample_every 10 --use_amp --output_dir outputs_ddpm6
```

The original job reached the 48 h limit during epoch 176. The thesis model is the
`best_model.pt` saved up to then (epoch 170, the lowest validation loss).

### What "reproducible" means here

All settings, the data split and the random seed (`--seed`, default 42) are fixed. GPU
kernels are not bitwise deterministic, so a retrained model matches the published one
statistically, not to the last bit. To reproduce the thesis figures exactly, use the
published weights.

Three behaviours are kept as they were in the thesis runs:

- On resume, the "best validation loss" starts from the validation loss of the resumed epoch,
  not from the best one before it. This only affects which epoch `best_model.pt` points to in
  job 2. The thesis uses `checkpoint_epoch_200.pt`, so it is unaffected.
- `losses.png` covers only the epochs run in the current job. Each epoch's losses are also
  printed to the Slurm log.
- Validation uses random timesteps and noise, as in training, so the validation loss of a
  fixed model varies slightly from run to run.

## 4. Generating maps and power spectra

```bash
# DDPM-2 at (Ωm, σ8) = (0.3, 0.8), 16 maps
python generate_map_pk.py --checkpoint weights/2param/model.pt --labels 0.3 0.8 --n_maps 16

# DDPM-6, fiducial astrophysics
python generate_map_pk.py --checkpoint weights/6param/model.pt --labels 0.3 0.8 1 1 1 1
```

`--sampler ddim` (the default, 50 steps) is the sampler used for the thesis figures. `--sampler
ancestral` runs the full 1500-step chain instead: about 30 times slower, but the most faithful
power spectrum (thesis Fig. 4.3 shows DDIM-50 under-producing small-scale power by 15–20 %).
`--eta 0` (the default) makes each DDIM map a deterministic function of `--seed`. Outputs go to
`outputs/<tag>/`: the maps in [0, 1] and in log₁₀ N_HI, `pk.npz`, `pk.csv`, `figure.png`
and `meta.json`. One map takes a few seconds on a GPU and about 40 s on a CPU.

P(k) is measured on the log₁₀ N_HI field: 25 Mpc/h box, 256 pixels, radial bins of width
2π/L up to the Nyquist frequency, k in h/Mpc. Model output x ∈ [0, 1] is converted with
log₁₀ N_HI = 14 + 8x. Because the training normalisation was per pixel (section 2), this
conversion is approximate.

## 5. Reproducing the thesis figures

`figures/` holds one script per figure in the results chapter. Each samples what it needs from the
released weights, caches it, and writes the figure and any tables:

```bash
sbatch slurm/figures.sh            # all of them, about 6.5 GPU-hours on one L40S
python figures/fig4_6_power_spectrum.py --out_dir results/fig4_6 --n_gen 300    # or one at a time
python figures/fig4_6_power_spectrum.py --out_dir results/fig4_6 --from_cache  # redraw on a CPU
```

`thesis_figures/README.md` lists the command for every figure and what each one measures. All
scripts accept `--sampler ancestral` to redo a figure with the full sampler. Re-run on the same
cached spectra, each script reproduces the thesis tables byte for byte and the figures pixel for
pixel. Freshly sampled maps are new realisations, so regenerated figures agree statistically.

## Training ranges

Ωₘ ∈ [0.1, 0.5], σ₈ ∈ [0.6, 1.0], A_SN1, A_AGN1 ∈ [0.25, 4], A_SN2, A_AGN2 ∈ [0.5, 2].
`generate_map_pk.py` warns when a value is outside these ranges.
