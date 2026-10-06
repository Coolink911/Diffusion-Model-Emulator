# Spearman: fidelity vs conditioning parameters — fidelity_metrics_ddpm6

n = 50 test points. Two-sided Spearman; 95% CI from 20000 bootstrap resamples of the pairs; q = Benjamini-Hochberg FDR within each metric (6 tests each).

**Minimum detectable |rho| at n = 50, alpha = 0.05: 0.28.** A non-significant result therefore does not establish insensitivity — read the CI.

| Metric | Parameter | rho | p | q (BH) | 95% CI on rho |
|:--|:--|--:|--:|--:|:--|
| r2_mu | Omega_m | +0.151 | 0.295 | 0.443 | [-0.16, +0.43] |
| r2_mu | sigma_8 | -0.201 | 0.161 | 0.443 | [-0.48, +0.11] |
| r2_mu | A_SN1 | +0.343 | 0.0147 | 0.088 | [+0.07, +0.57] |
| r2_mu | A_AGN1 | -0.090 | 0.534 | 0.636 | [-0.37, +0.21] |
| r2_mu | A_SN2 | +0.069 | 0.636 | 0.636 | [-0.19, +0.32] |
| r2_mu | A_AGN2 | -0.169 | 0.241 | 0.443 | [-0.44, +0.13] |
| r2_sig | Omega_m | +0.396 ** | 0.00446 | 0.0134 | [+0.15, +0.59] |
| r2_sig | sigma_8 | -0.181 | 0.207 | 0.25 | [-0.45, +0.11] |
| r2_sig | A_SN1 | +0.191 | 0.184 | 0.25 | [-0.10, +0.46] |
| r2_sig | A_AGN1 | +0.496 ** | 0.000249 | 0.00149 | [+0.25, +0.69] |
| r2_sig | A_SN2 | +0.181 | 0.208 | 0.25 | [-0.12, +0.46] |
| r2_sig | A_AGN2 | -0.117 | 0.419 | 0.419 | [-0.40, +0.19] |
| dex_mu | Omega_m | -0.286 | 0.044 | 0.195 | [-0.54, +0.01] |
| dex_mu | sigma_8 | +0.189 | 0.189 | 0.378 | [-0.10, +0.45] |
| dex_mu | A_SN1 | +0.085 | 0.557 | 0.647 | [-0.21, +0.36] |
| dex_mu | A_AGN1 | -0.263 | 0.0649 | 0.195 | [-0.47, -0.01] |
| dex_mu | A_SN2 | +0.066 | 0.647 | 0.647 | [-0.23, +0.35] |
| dex_mu | A_AGN2 | +0.134 | 0.354 | 0.531 | [-0.19, +0.43] |
| dex_sig | Omega_m | -0.128 | 0.375 | 0.606 | [-0.39, +0.17] |
| dex_sig | sigma_8 | -0.050 | 0.733 | 0.733 | [-0.33, +0.25] |
| dex_sig | A_SN1 | -0.096 | 0.505 | 0.606 | [-0.37, +0.19] |
| dex_sig | A_AGN1 | -0.102 | 0.479 | 0.606 | [-0.39, +0.20] |
| dex_sig | A_SN2 | -0.147 | 0.307 | 0.606 | [-0.40, +0.13] |
| dex_sig | A_AGN2 | -0.098 | 0.499 | 0.606 | [-0.35, +0.17] |

## Reading

- **r2_mu** (raw-P R^2 (dominated by 3 low-k modes)): no parameter significant after FDR. Widest CI is sigma_8 at [-0.48, +0.11], so correlations up to |rho| ~ 0.48 remain compatible with this sample.
- **r2_sig** (raw-P R^2 (dominated by 3 low-k modes)): 2 of 6 parameters significant after FDR (Omega_m, A_AGN1). Widest CI is A_AGN2 at [-0.40, +0.19], so correlations up to |rho| ~ 0.40 remain compatible with this sample.
- **dex_mu** (dex (preferred)): no parameter significant after FDR. Widest CI is A_AGN2 at [-0.19, +0.43], so correlations up to |rho| ~ 0.43 remain compatible with this sample.
- **dex_sig** (dex (preferred)): no parameter significant after FDR. Widest CI is A_AGN1 at [-0.39, +0.20], so correlations up to |rho| ~ 0.39 remain compatible with this sample.
