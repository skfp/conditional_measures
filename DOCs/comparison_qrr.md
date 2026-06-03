# QRR vs Proposed Methods: simulation comparison

Comparison of Farcomeni & Geraci (2024) QRR (`Qtools::qrr()`) against the
proposed IOQR and IAQR estimators. All MSE values are ×10³.
QRR uses the R package implementation directly via rpy2.
See `mc_fld.py::estimate_indices_qrr` for details.

## 1. Well-specified DGP (EFLD, log-linear CQF)

Source: `mse_260601_qrr/`. Nine (β, c) parameter sets, n ∈ {50, 100},
1000 MC replications each.

### 1a. Overall mean MSE (averaged over all parameter sets and x values)

**n = 50**

| method | qDI | qZI |
| --- | --- | --- |
| IOQR | 2.486 | 2.381 |
| IAQR | 2.137 | 2.222 |
| BK | 2.563 | 2.685 |
| BRW | 2.519 | 2.559 |
| WL1 | 2.448 | 2.43 |
| QRR | 3.327 | 3.158 |

**n = 100**

| method | qDI | qZI |
| --- | --- | --- |
| IOQR | 1.244 | 1.172 |
| IAQR | 1.12 | 1.129 |
| BK | 1.263 | 1.219 |
| BRW | 1.231 | 1.203 |
| WL1 | 1.203 | 1.179 |
| QRR | 1.615 | 1.443 |

### 1b. QRR / IOQR and QRR / IAQR ratios by inequality level (β)

Median ratio and share of (x, index, param set) cases where QRR has lower MSE.

**n = 50**

| beta | QRR/IOQR median | QRR/IOQR wins | QRR/IAQR median | QRR/IAQR wins |
| --- | --- | --- | --- | --- |
| 0.05 | 1.02 | 50% | 0.94 | 57% |
| 0.1 | 1.24 | 0% | 1.23 | 0% |
| 0.2 | 1.26 | 14% | 1.36 | 14% |
| 0.5 | 1.9 | 0% | 2.44 | 0% |

**n = 100**

| beta | QRR/IOQR median | QRR/IOQR wins | QRR/IAQR median | QRR/IAQR wins |
| --- | --- | --- | --- | --- |
| 0.05 | 1.01 | 50% | 0.96 | 64% |
| 0.1 | 1.31 | 0% | 1.3 | 0% |
| 0.2 | 1.19 | 7% | 1.23 | 7% |
| 0.5 | 1.43 | 14% | 1.61 | 0% |

### 1c. Mean QRR / IOQR ratio by x value (n = 50, all parameter sets)

| x | QRR/IOQR | QRR/IAQR |
| --- | --- | --- |
| 1 | 1.0 | 1.09 |
| 5 | 1.48 | 1.68 |
| 10 | 2.16 | 2.62 |
| 15 | 1.77 | 2.17 |
| 20 | 1.26 | 1.47 |
| 25 | 1.27 | 1.42 |
| 30 | 1.56 | 1.73 |

### 1d. Parameter sets where QRR is worst (n = 50, qDI)

| index | beta | c | x | ratio |
| --- | --- | --- | --- | --- |
| 0 | 0.5 | 0.5 | 10 | 6.57 |
| 1 | 0.5 | 0.5 | 15 | 5.48 |
| 2 | 0.5 | 0.5 | 5 | 3.66 |
| 3 | 0.2 | 1.0 | 10 | 3.01 |
| 4 | 0.5 | 0.5 | 20 | 2.85 |
| 5 | 0.2 | 1.0 | 15 | 2.36 |
| 6 | 0.1 | 1.0 | 10 | 2.28 |
| 7 | 0.1 | 0.1 | 30 | 1.95 |

### Summary (well-specified)

Under the well-specified EFLD model (n=50), QRR has mean qDI MSE of 3.327 vs 2.486 for IOQR and 2.137 for IAQR. The QRR/IOQR ratio has median 1.13 and QRR achieves lower MSE in only 29% of (x, β, c, index) combinations. QRR is most competitive at low inequality (β=0.05) and deteriorates sharply at high inequality (β=0.5), where outcome-level weighting amplifies the influence of extreme observations.

## 2. Misspecified DGPs

Source: `mse_misspec/`. Three DGPs, n ∈ {50, 100}, 1000 MC replications.

### 2a. Mean MSE by DGP (n = 50, averaged over x)

**DGP 3a: quadratic CQF (c₂·p·x² term)**

| method | qDI | qZI |
| --- | --- | --- |
| IOQR | 3.283 | 3.072 |
| IAQR | 2.731 | 2.712 |
| BK | 3.356 | 3.437 |
| BRW | 3.292 | 3.233 |
| WL1 | 3.227 | 3.015 |
| QRR | 3.499 | 3.232 |

**DGP 3b: log-normal heteroscedastic (non-EFLD marginal)**

| method | qDI | qZI |
| --- | --- | --- |
| IOQR | 2.801 | 3.317 |
| IAQR | 2.616 | 3.273 |
| BK | 2.773 | 3.472 |
| BRW | 2.785 | 3.446 |
| WL1 | 2.836 | 3.596 |
| QRR | 3.078 | 3.745 |

**DGP 3c: non-monotone β₁ (bell-shaped)**

| method | qDI | qZI |
| --- | --- | --- |
| IOQR | 4.153 | 2.154 |
| IAQR | 3.197 | 1.866 |
| BK | 4.165 | 3.341 |
| BRW | 4.22 | 3.11 |
| WL1 | 3.7 | 2.674 |
| QRR | 4.613 | 4.698 |

### 2b. QRR / IOQR and QRR / IAQR ratios by DGP

**n = 50**

| dgp | QRR/IOQR median | QRR/IOQR wins | QRR/IAQR median | QRR/IAQR wins |
| --- | --- | --- | --- | --- |
| 3a | 1.1 | 29% | 1.28 | 7% |
| 3b | 1.11 | 14% | 1.18 | 14% |
| 3c | 1.64 | 21% | 2.08 | 0% |

**n = 100**

| dgp | QRR/IOQR median | QRR/IOQR wins | QRR/IAQR median | QRR/IAQR wins |
| --- | --- | --- | --- | --- |
| 3a | 1.05 | 43% | 1.12 | 29% |
| 3b | 1.15 | 7% | 1.16 | 7% |
| 3c | 1.3 | 36% | 1.35 | 29% |

### 2c. Mean QRR / IOQR ratio by x value, per DGP (n = 50)

**DGP 3a**

| x | QRR/IOQR | QRR/IAQR |
| --- | --- | --- |
| 1 | 0.81 | 1.04 |
| 5 | 1.43 | 1.49 |
| 10 | 1.67 | 1.6 |
| 15 | 1.23 | 1.34 |
| 20 | 1.05 | 1.21 |
| 25 | 1.01 | 1.17 |
| 30 | 1.02 | 1.17 |

**DGP 3b**

| x | QRR/IOQR | QRR/IAQR |
| --- | --- | --- |
| 1 | 0.91 | 0.91 |
| 5 | 1.1 | 1.08 |
| 10 | 1.51 | 1.52 |
| 15 | 1.28 | 1.36 |
| 20 | 1.07 | 1.16 |
| 25 | 1.09 | 1.18 |
| 30 | 1.21 | 1.3 |

**DGP 3c**

| x | QRR/IOQR | QRR/IAQR |
| --- | --- | --- |
| 1 | 1.33 | 1.82 |
| 5 | 1.62 | 2.15 |
| 10 | 1.9 | 2.34 |
| 15 | 1.98 | 2.19 |
| 20 | 1.58 | 1.64 |
| 25 | 1.78 | 1.8 |
| 30 | 2.27 | 2.3 |

### Summary (misspecified)

- **DGP 3a (quadratic CQF (c₂·p·x² term))**: QRR/IOQR median 1.10, QRR wins in 29% of cases, worst ratio 1.7×.
- **DGP 3b (log-normal heteroscedastic (non-EFLD marginal))**: QRR/IOQR median 1.11, QRR wins in 14% of cases, worst ratio 1.6×.
- **DGP 3c (non-monotone β₁ (bell-shaped))**: QRR/IOQR median 1.64, QRR wins in 21% of cases, worst ratio 3.1×.

In all three misspecification scenarios QRR is outperformed by IOQR and IAQR.
The largest gap is in DGP 3c (non-monotone β₁) where QRR's outcome-level
weighting is most harmful at large x. DGP 3b (log-normal) shows the smallest
gap, reflecting mild departure from the log-linear model.