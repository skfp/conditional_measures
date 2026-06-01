# Extension plan: simulations paper

## Context

The original manuscript (arXiv:2412.20228) was rejected by STPA. We are splitting it into two papers:
a theory paper and a simulations paper. This document covers the plan for the **simulations paper**.

The reviewer's simulation-relevant criticisms are:
- **#3** — no comparison with simpler baselines (stratified Gini, inter-quantile ratios, etc.)
- **#5** — quantile ratio regression (QRR, Farcomeni & Geraci 2024) must be included as a benchmark
- **#8** — no bootstrap, no confidence intervals, no coverage assessment
- **#9** — DGP always well-specified; bias–variance trade-off of isotonic smoothing not explored; no misspecification scenarios
- **#10** — evaluation should explicitly check accuracy at the p/2 and 1−p/2 quantile levels that enter qZ/qD

---

## What already exists (keep)

- **Methods**: IOQR, IAQR, BK (Bassett-Koenker), BRW (Bondell-Reich-Wang), WL1 (Wu-Liu), CQR (Koenker-Ng via R)
- **DGP**: EFLD(α, β, c) with κ = cx, perfectly log-linear and well-specified
- **Metrics**: MSE, error boxplots
- **Grid**: n ∈ {50, 100, 500, 1000}, 9 parameter combos, x ∈ {1, …, 30}, 1000 MC repetitions

---

## Extension 1 — Add QRR as a direct benchmark (reviewer #4, #5)

**Priority: highest.** The reviewer calls QRR "a natural benchmark" and says its omission "weakens the empirical and methodological contribution." Since qZ and qD are themselves ratios of quantiles, QRR (Farcomeni & Geraci 2024) directly targets the same quantities and cannot be left out.

**What to implement**: wrap `qrr::qrr()` via rpy2, the same way CQR wraps `quantreg::rq()`. For each x value, recover qZI and qDI from the fitted quantile ratio model and feed into the existing evaluation pipeline.

**Expected story**: QRR should perform well in the well-specified setting. The interesting contrast will appear in misspecification scenarios (Extension 3).

---

## Extension 2 — Add simpler baselines (reviewer #3)

The reviewer asks whether the full machinery yields anything over simpler tools that a practitioner would reach for first.

**2a. Stratified empirical estimator**: Bin x into K groups (e.g., K ∈ {5, 10}), compute empirical qZI/qDI within each bin, smooth across bins (running mean or isotonic regression). Simple but crude; performance degrades with bin width.

**2b. Inter-quantile-ratio regression (IQRR)**: Directly regress log(Q̂(0.75)/Q̂(0.25)) or log(Q̂(0.9)/Q̂(0.1)) on x as a scalar proxy for conditional inequality. The simplest thing a practitioner would do. Computationally trivial and serves as a straw-man baseline that the proposed methods should clearly beat.

---

## Extension 3 — Misspecification scenarios (reviewer #9)

Currently the DGP is always perfectly log-linear, which favors all quantile-regression-based methods, especially IOQR and IAQR. Three new settings:

**3a. Nonlinear conditional quantile function**: e.g., Q_x(p) = exp(β₀(p) + β₁(p)·x + β₂(p)·x²). All methods are fitted with the linear model. Tests robustness to a broken linearity assumption.

**3b. Non-EFLD marginal**: e.g., Y|X=x ~ Weibull or log-normal with scale depending on x. The quantile function is still valid and monotone in p, but has a different shape from EFLD. Tests whether the log-linear interpolation assumption causes meaningful distortion in practice.

**3c. Non-monotone β₁(p)**: A DGP where Q_x(p) is non-decreasing in p for each x (so the conditional quantile function is valid), but the individual coefficient β₁(p) is not monotone in p. This directly tests reviewer comment #6 — isotonic correction on individual βᵢ is not necessary and may hurt when β₁ is genuinely non-monotone. This scenario is the most important of the three: it exposes the theoretical weakness flagged by the reviewer and lets the simulation show empirically how much IOQR and IAQR lose in that setting.

---

## Extension 4 — Bias–variance decomposition (reviewer #9)

Replace or supplement the MSE heatmaps with decomposed bias² + variance plots. This directly addresses the reviewer's concern that "isotonic regression reduces variance but may introduce bias — this trade-off is not explored."

**No new simulation code required** — decompose from the existing MC output:
- bias(x) = mean(estimate) − true_value
- variance(x) = var(estimate)
- MSE = bias² + variance

---

## Extension 5 — Bootstrap confidence intervals and coverage (reviewer #8)

Currently there is no uncertainty quantification. Add a bootstrap simulation study for a subset of scenarios (e.g., 3 parameter combos × 3 sample sizes):

- For each MC replicate, draw B = 500 bootstrap samples and compute qZI(Q_x) and qDI(Q_x) for each
- Report: 95% pointwise bootstrap CI, empirical coverage rate (fraction of CIs containing the true value), CI width as a function of x and n

**Computational note**: 500 bootstrap × 1000 MC = 500k estimator calls per scenario. Implement as a separate script and run only on a subset of scenarios. This is the most expensive extension; implement last.

---

## Extension 6 — Accuracy at p/2 and 1−p/2 quantile levels (reviewer #10)

Currently evaluation is at the level of the integrated indices qZI/qDI. The reviewer asks to verify accuracy specifically at the intermediate quantile levels that enter those integrals: Q̂(p/2) and Q̂(1−p/2) for representative p.

**What to add**: for each method and each x, report RMSE of Q̂(u) at u ∈ {0.05, 0.1, 0.25, 0.5} — i.e., the p/2 range relevant to qZ/qD for p up to 0.5. Checks whether estimation error at low quantile levels propagates into the final index.

---

## Revised paper structure

| Section | Content |
|---|---|
| 1. Introduction | What conditional inequality measures are, why existing tools are insufficient, preview of findings |
| 2. Methods | Brief descriptions of all estimators: IOQR, IAQR, BK, BRW, WL1, CQR, QRR, simple baselines |
| 3. Simulation — well-specified | Extended existing results + bias/variance decomposition (Extensions 1, 2, 4) |
| 4. Simulation — misspecification | New DGPs (Extension 3); story of how each method degrades |
| 5. Accuracy at tail quantiles | Diagnostic at p/2 and 1−p/2 levels (Extension 6) |
| 6. Bootstrap uncertainty | Coverage and CI width study (Extension 5) |
| 7. Real data | Census 2000 analysis, extended, possibly with bootstrap CIs |
| 8. Conclusion | Summary table of method pros/cons across all scenarios |

---

## Priority ordering

| Priority | Extension | Effort | Reviewer comment |
|---|---|---|---|
| 1 | QRR benchmark | Medium (rpy2 wrapper) | #4, #5 |
| 2 | Misspecification 3c (non-monotone β₁) | Medium (new DGP) | #6, #9 |
| 3 | Bias–variance decomposition | Low (recompute from existing output) | #9 |
| 4 | Misspecification 3a/3b | Medium (new DGP code) | #9 |
| 5 | Simple baselines (stratified, IQRR) | Low–medium | #3 |
| 6 | Accuracy at p/2, 1−p/2 levels | Low (new diagnostic) | #10 |
| 7 | Bootstrap coverage | High (compute-intensive) | #8 |
