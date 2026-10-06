# Spearman: fidelity vs conditioning parameters — fidelity_metrics_ddpm2

n = 50 test points. Two-sided Spearman; 95% CI from 20000 bootstrap resamples of the pairs; q = Benjamini-Hochberg FDR within each metric (2 tests each).

**Minimum detectable |rho| at n = 50, alpha = 0.05: 0.28.** A non-significant result therefore does not establish insensitivity — read the CI.

| Metric | Parameter | rho | p | q (BH) | 95% CI on rho |
|:--|:--|--:|--:|--:|:--|
| r2_mu | Omega_m | -0.015 | 0.918 | 0.918 | [-0.30, +0.26] |
| r2_mu | sigma_8 | -0.159 | 0.27 | 0.541 | [-0.43, +0.14] |
| r2_sig | Omega_m | +0.252 | 0.077 | 0.077 | [-0.01, +0.49] |
| r2_sig | sigma_8 | -0.329 ** | 0.0195 | 0.0389 | [-0.58, -0.05] |
| dex_mu | Omega_m | +0.049 | 0.737 | 0.737 | [-0.22, +0.30] |
| dex_mu | sigma_8 | +0.201 | 0.162 | 0.324 | [-0.10, +0.48] |
| dex_sig | Omega_m | -0.316 | 0.0252 | 0.0504 | [-0.57, -0.03] |
| dex_sig | sigma_8 | +0.203 | 0.157 | 0.157 | [-0.10, +0.48] |

## Reading

- **r2_mu** (raw-P R^2 (dominated by 3 low-k modes)): no parameter significant after FDR. Widest CI is sigma_8 at [-0.43, +0.14], so correlations up to |rho| ~ 0.43 remain compatible with this sample.
- **r2_sig** (raw-P R^2 (dominated by 3 low-k modes)): 1 of 2 parameters significant after FDR (sigma_8). Widest CI is sigma_8 at [-0.58, -0.05], so correlations up to |rho| ~ 0.58 remain compatible with this sample.
- **dex_mu** (dex (preferred)): no parameter significant after FDR. Widest CI is sigma_8 at [-0.10, +0.48], so correlations up to |rho| ~ 0.48 remain compatible with this sample.
- **dex_sig** (dex (preferred)): no parameter significant after FDR. Widest CI is sigma_8 at [-0.10, +0.48], so correlations up to |rho| ~ 0.48 remain compatible with this sample.
