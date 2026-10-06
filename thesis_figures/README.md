# Results-chapter figures (revised October 2026)

Final versions of the Chapter 4 figures and tables, regenerated in response to the July 2026
review. Each folder holds the figure (PNG, and PDF where produced) plus any table or caption
generated with it.

| Folder | Thesis item | What changed from the earlier draft |
|---|---|---|
| `fig4.2_training_curves/` | Fig. 4.2 | The DDPM-2 curve is digitised from the `losses.png` files written during training (no log survives). The earlier digitisation traced the legend box as data, which put a false spike at epochs 141–148. Those epochs were re-read from the curve pixels. |
| `fig4.3_sampler_study/` | Fig. 4.3 | Timing and fidelity both measured at 256² on the same 50 test labels and starting noise. Fidelity is the RMS of log₁₀(⟨P_gen⟩/⟨P_CAMELS⟩) in dex with a bootstrap s.d., replacing R² on raw P(k). Adds DDIM-50 at η = 1. |
| `fig4.4_4.5_map_grids/` | Figs. 4.4, 4.5 | Shared colour bar in log₁₀ N_HI [cm⁻²] and a 5 h⁻¹ Mpc scale bar. All six parameters labelled. |
| `fig4.6_tab4.4_power_spectrum/` | Fig. 4.6, Table 4.4 | CAMELS reference = the 15 maps at exactly each θ (previously 15 nearest neighbours). 300 generated maps per anchor (previously 15). Ratio panels in physical k. Table reports R² on log₁₀ P and the median fractional residual, with bootstrap s.d. |
| `fig4.8_fidelity_map/` | Fig. 4.8 | Points coloured by the dex mismatch of μ(P) and σ(P) instead of R² on raw P(k), with Spearman correlations against each parameter. |
| `fig4.10_pixel_pdf/` | Fig. 4.10 | 300 generated maps per anchor, all plotted. Row labels, axis labels and measured peak positions; `pdf_peak_shifts.md` lists the shifts. |
| `fig4.11_sensitivity/` | Fig. 4.11 | The x-axis is converted as log₁₀ N_HI = 14 + 8x (the first version used 14 + 7x, shifting every curve). Panel titles give the envelope for DDPM-6 and, for Ωₘ and σ₈, for the CAMELS 1P simulations. The suptitle gives the envelope formula and the Monte-Carlo noise floor. |

## Producing scripts

Each figure is made by one script in `../figures/`. Run them from the repository root after
`prepare_data.py` (CAMELS splits in `data/`) with the released weights in `weights/2param/` and
`weights/6param/`, or run them all with `sbatch slurm/figures.sh` (about 6.5 GPU-hours).

| Figure | Command |
|---|---|
| 4.2 | `python figures/fig4_2_training_curves.py --out results/fig4_2/training_val_overlay.png` |
| 4.3 | `python figures/fig4_3_sampler_study.py --out_dir results/fig4_3` |
| 4.4, 4.5 | `python figures/fig4_4_4_5_map_grids.py --out_dir results/fig4_4_4_5` |
| 4.6, Table 4.4 | `python figures/fig4_6_power_spectrum.py --out_dir results/fig4_6 --n_gen 300` |
| 4.8 | `python figures/fig4_8_fidelity_map.py --model ddpm2 --out_dir results/fig4_8` (and `--model ddpm6`) |
| 4.10 | `python figures/fig4_10_pixel_pdf.py --out_dir results/fig4_10 --n_gen 300 --display_smooth 1.5` |
| 4.11 | `python figures/fig4_11_sensitivity.py --out_dir results/fig4_11` (add `--camels_1p_maps`/`--camels_1p_params` for the CAMELS envelopes) |

Each script caches what it samples, and `--from_cache` (`--plot_only` for 4.3) redraws the figure
and tables on a CPU. Every script was checked against the outputs in this folder: re-run from the
same cached spectra or maps, it reproduces the tables byte for byte and the figures pixel for
pixel, and its sampling is bitwise identical to the code that made them. Newly generated maps will
differ realisation by realisation (GPU sampling is not bitwise deterministic), so regenerated
figures agree statistically rather than exactly.

## Caveats that apply to these figures

- **Sampler.** All DDPM-2/DDPM-6 maps are drawn with DDIM-50 (η = 0, or η = 1 in Fig. 4.11),
  the thesis operating point. Fig. 4.3 shows DDIM-50 under-produces power by 15–20 % at
  k ≈ 3–20 h/Mpc relative to the full ancestral sampler. Part of the high-k deficit in
  Fig. 4.6 is therefore due to the sampler rather than the model.
- **Test set.** The train/val/test split is per map, not per cosmology. Every test cosmology
  also has maps in the training set, so "test cosmologies" are not held out.
- **Units.** log₁₀ N_HI = 14 + 8x is an approximation: the training maps were min-max
  normalised separately at each pixel position.
- **P(k) amplitude.** P(k) in Figs. 4.6 and 4.8 is computed on the [0, 1] map field, so its
  amplitude is 1/64 of the log₁₀ N_HI power spectrum. Ratios are unaffected.
- **Not included.** The HIGlow benchmark figure is absent. The 64² data HIGlow was trained
  on has images shuffled relative to their labels, so that comparison is not valid.
