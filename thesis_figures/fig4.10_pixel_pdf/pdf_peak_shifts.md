# PDF peak shifts per label set (measured, not eyeballed)

CAMELS = 15 maps at exactly the anchor's theta; 300 maps generated per model. Peaks by parabolic interpolation about the arg-max on curves smoothed with a 1-bin Gaussian; bin width 0.080 dex. All shifts in dex, generated minus CAMELS.

| Row | Ωm | σ8 | Model | μ peak CAMELS | μ peak gen | **μ shift** | σ peak CAMELS | σ peak gen | **σ shift** | mean shift | median shift | W₁(μ) |
|---:|---:|---:|:--|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 0.329 | 0.750 | DDPM-2 | 15.932 | 16.088 | **+0.156** | 15.381 | 15.491 | **+0.110** | +0.160 | +0.156 | 0.161 |
| 1 | 0.329 | 0.750 | DDPM-6 | 15.932 | 16.100 | **+0.168** | 15.381 | 15.482 | **+0.101** | +0.153 | +0.153 | 0.153 |
| 2 | 0.282 | 0.724 | DDPM-2 | 16.101 | 16.160 | **+0.059** | 15.558 | 15.600 | **+0.042** | +0.022 | +0.037 | 0.033 |
| 2 | 0.282 | 0.724 | DDPM-6 | 16.101 | 16.208 | **+0.107** | 15.558 | 15.596 | **+0.038** | +0.061 | +0.075 | 0.065 |
| 3 | 0.161 | 0.853 | DDPM-2 | 16.118 | 16.176 | **+0.059** | 15.631 | 15.656 | **+0.026** | -0.011 | +0.015 | 0.026 |
| 3 | 0.161 | 0.853 | DDPM-6 | 16.118 | 16.222 | **+0.104** | 15.631 | 15.693 | **+0.062** | +0.037 | +0.057 | 0.047 |
| 4 | 0.105 | 0.759 | DDPM-2 | 16.179 | 16.285 | **+0.105** | 15.859 | 15.858 | **-0.001** | +0.034 | +0.057 | 0.045 |
| 4 | 0.105 | 0.759 | DDPM-6 | 16.179 | 16.332 | **+0.153** | 15.859 | 15.902 | **+0.044** | +0.081 | +0.103 | 0.082 |
| 5 | 0.201 | 0.747 | DDPM-2 | 15.997 | 16.217 | **+0.220** | 15.523 | 15.662 | **+0.140** | +0.149 | +0.173 | 0.154 |
| 5 | 0.201 | 0.747 | DDPM-6 | 15.997 | 16.208 | **+0.211** | 15.523 | 15.664 | **+0.142** | +0.138 | +0.159 | 0.139 |
| 6 | 0.247 | 0.607 | DDPM-2 | 16.211 | 16.318 | **+0.107** | 15.764 | 15.800 | **+0.036** | +0.067 | +0.080 | 0.070 |
| 6 | 0.247 | 0.607 | DDPM-6 | 16.211 | 16.290 | **+0.079** | 15.764 | 15.763 | **-0.002** | +0.047 | +0.060 | 0.052 |

## R² per subplot

`R²(support)` uses the 90 bins where CAMELS carries signal (>10⁻³ of peak); `R²(all bins)` includes the empty tail, which both curves match for free and which therefore inflates the score. `matched N=15` subsamples the generated maps down to the CAMELS map count (400 draws) so the two σ estimates carry the same sampling noise — CAMELS LH stores only 15 maps per parameter set, so the ensembles cannot be matched by generating more.

| Row | Ωm | σ8 | Model | R²(μ) support | R²(μ) all bins | R²(σ) support | R²(σ) all bins | R²(μ) matched N=15 | R²(σ) matched N=15 |
|---:|---:|---:|:--|---:|---:|---:|---:|:--|:--|
| 1 | 0.329 | 0.750 | DDPM-2 | 0.941 | 0.943 | 0.859 | 0.866 | 0.934 ± 0.028 | 0.774 ± 0.198 |
| 1 | 0.329 | 0.750 | DDPM-6 | 0.944 | 0.946 | 0.870 | 0.876 | 0.940 ± 0.029 | 0.801 ± 0.125 |
| 2 | 0.282 | 0.724 | DDPM-2 | 0.992 | 0.993 | 0.955 | 0.957 | 0.987 ± 0.012 | 0.901 ± 0.159 |
| 2 | 0.282 | 0.724 | DDPM-6 | 0.984 | 0.985 | 0.886 | 0.892 | 0.977 ± 0.020 | 0.835 ± 0.132 |
| 3 | 0.161 | 0.853 | DDPM-2 | 0.998 | 0.998 | 0.971 | 0.972 | 0.992 ± 0.010 | 0.904 ± 0.117 |
| 3 | 0.161 | 0.853 | DDPM-6 | 0.990 | 0.991 | 0.787 | 0.796 | 0.982 ± 0.019 | 0.710 ± 0.298 |
| 4 | 0.105 | 0.759 | DDPM-2 | 0.985 | 0.986 | 0.872 | 0.882 | 0.979 ± 0.017 | 0.771 ± 0.111 |
| 4 | 0.105 | 0.759 | DDPM-6 | 0.969 | 0.970 | 0.924 | 0.930 | 0.959 ± 0.033 | 0.854 ± 0.084 |
| 5 | 0.201 | 0.747 | DDPM-2 | 0.918 | 0.921 | 0.870 | 0.874 | 0.912 ± 0.039 | 0.805 ± 0.094 |
| 5 | 0.201 | 0.747 | DDPM-6 | 0.934 | 0.936 | 0.873 | 0.878 | 0.929 ± 0.035 | 0.823 ± 0.087 |
| 6 | 0.247 | 0.607 | DDPM-2 | 0.982 | 0.983 | 0.943 | 0.946 | 0.977 ± 0.017 | 0.872 ± 0.137 |
| 6 | 0.247 | 0.607 | DDPM-6 | 0.988 | 0.989 | 0.812 | 0.823 | 0.983 ± 0.015 | 0.777 ± 0.284 |

## σ(PDF) is bimodal — all modes above 25% of the peak

| Row | Model | CAMELS modes | generated modes |
|---:|:--|:--|:--|
| 1 | DDPM-2 | 15.38, 16.28 | 15.49, 16.51 |
| 1 | DDPM-6 | 15.38, 16.28 | 15.48, 16.51 |
| 2 | DDPM-2 | 15.56, 16.50 | 15.60, 16.54 |
| 2 | DDPM-6 | 15.56, 16.50 | 15.60, 16.63 |
| 3 | DDPM-2 | 15.63, 16.60 | 15.66, 16.65 |
| 3 | DDPM-6 | 15.63, 16.60 | 15.69, 16.74 |
| 4 | DDPM-2 | 15.86, 16.79 | 15.86, 16.77 |
| 4 | DDPM-6 | 15.86, 16.79 | 15.90, 16.95 |
| 5 | DDPM-2 | 15.52, 16.51 | 15.66, 16.64 |
| 5 | DDPM-6 | 15.52, 16.51 | 15.66, 16.67 |
| 6 | DDPM-2 | 15.76, 16.59 | 15.80, 16.73 |
| 6 | DDPM-6 | 15.76, 16.59 | 15.76, 16.78 |

**DDPM-2** — μ peak shift +0.118 dex mean, 0.220 dex worst; σ peak shift +0.059 dex mean, 0.140 dex worst.

**DDPM-6** — μ peak shift +0.137 dex mean, 0.211 dex worst; σ peak shift +0.064 dex mean, 0.142 dex worst.

## Envelope score

```
E = ∫ [ max_v μ_v(L) − min_v μ_v(L) ] dL
  ≈ Σ_b [ max_v μ_v(L_b) − min_v μ_v(L_b) ] · ΔL
```

with L = log10 N_HI integrated over the full support [14, 22], each μ_v density-normalised so ∫μ_v dL = 1, and v running over the sampled values of the parameter. E is dimensionless, bounded in [0, 2] (0 = identical curves, 2 = disjoint support), and invariant under the linear remap between normalised pixel value and dex, so the [0,1]-space and dex-space values coincide. As implemented in `fig4_11_sensitivity.py::sensitivity_scores` the sum uses 80 uniform bins.
