# Six-anchor P(k) agreement (replaces the raw-P R² table)

Fractional residual ⟨P_gen/P_CAMELS − 1⟩ by k band [h/Mpc]; deviation in dex; R² on log₁₀P as the secondary scalar. `r2_rawP` is retained only to document that R² on raw P(k) is insensitive to small-scale error. ± on mean|Δ| is the bootstrap standard deviation (1000 resamples of both the CAMELS and generated maps).

| Anchor | Ωm | σ8 | Model | 0.25–1 | 1–8 | 8–20 | 20–32 | mean\|Δ\| dex ± boot. | mean\|Δ\| dex (k<8) | max\|Δ\| dex | R²(log₁₀P) | R²(raw P) |
|---:|---:|---:|:--|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 0.329 | 0.750 | DDPM-2 | -25.1% | -2.5% | -2.2% | +3.6% | 0.017 ± 0.010 | 0.026 | 0.153 | 0.9988 | 0.93472 |
| 0 | 0.329 | 0.750 | DDPM-6 | -29.6% | -2.0% | +8.1% | +21.5% | 0.052 ± 0.020 | 0.031 | 0.173 | 0.9936 | 0.89401 |
| 1 | 0.282 | 0.724 | DDPM-2 | +0.8% | -13.1% | -14.2% | -5.7% | 0.049 ± 0.018 | 0.058 | 0.095 | 0.9952 | 0.99150 |
| 1 | 0.282 | 0.724 | DDPM-6 | +2.8% | -10.1% | -5.0% | +11.2% | 0.038 ± 0.007 | 0.045 | 0.083 | 0.9970 | 0.99378 |
| 2 | 0.161 | 0.853 | DDPM-2 | -8.9% | -20.2% | -19.1% | +6.4% | 0.073 ± 0.011 | 0.095 | 0.153 | 0.9908 | 0.98037 |
| 2 | 0.161 | 0.853 | DDPM-6 | -3.3% | -12.6% | -5.9% | +22.4% | 0.060 ± 0.008 | 0.058 | 0.143 | 0.9937 | 0.98991 |
| 3 | 0.105 | 0.759 | DDPM-2 | +1.3% | -14.3% | -8.9% | +27.9% | 0.072 ± 0.014 | 0.063 | 0.185 | 0.9919 | 0.99183 |
| 3 | 0.105 | 0.759 | DDPM-6 | +2.5% | -10.0% | +15.9% | +91.2% | 0.142 ± 0.031 | 0.045 | 0.382 | 0.9620 | 0.98943 |
| 4 | 0.201 | 0.747 | DDPM-2 | -3.8% | -10.8% | -10.9% | +1.8% | 0.039 ± 0.023 | 0.049 | 0.099 | 0.9970 | 0.99886 |
| 4 | 0.201 | 0.747 | DDPM-6 | -10.0% | -4.3% | +7.2% | +25.3% | 0.056 ± 0.032 | 0.029 | 0.148 | 0.9931 | 0.98696 |
| 5 | 0.247 | 0.607 | DDPM-2 | +3.0% | -12.5% | -8.1% | +3.8% | 0.036 ± 0.009 | 0.056 | 0.085 | 0.9972 | 0.99120 |
| 5 | 0.247 | 0.607 | DDPM-6 | +2.0% | -11.7% | -3.4% | +15.0% | 0.043 ± 0.010 | 0.051 | 0.105 | 0.9958 | 0.99729 |

**DDPM-2 across the six anchors** — mean |Δ| = 0.048 dex (worst anchor 0.073); restricted to k < 8 h/Mpc, comparable to the 64² baselines in §4.1: 0.058 dex; R²(log₁₀P) mean 0.9952, min 0.9908; R²(raw P) min 0.93472.

**DDPM-6 across the six anchors** — mean |Δ| = 0.065 dex (worst anchor 0.142); restricted to k < 8 h/Mpc, comparable to the 64² baselines in §4.1: 0.043 dex; R²(log₁₀P) mean 0.9892, min 0.9620; R²(raw P) min 0.89401.

Reference (§4.1, 64² baselines, k < 8.04 h/Mpc): HIGlow 0.029 dex, HIDM 0.122 dex.
