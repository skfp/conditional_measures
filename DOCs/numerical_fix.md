# Numerical fix: QRR_linear implementation

## Problem

The original QRR_linear estimator used Nelder-Mead to minimise the nonlinear check function
on the Y-scale:

    min_{a,b} Σᵢ ρ_τ(exp(yᵢ) − exp(a + b·xᵢ))

This objective is **non-convex** in (a, b). For extreme quantile levels (τ near 0 or 1),
Nelder-Mead drifted to |b| values of order 1e10. These then fed into the qZ/qD ratio
computation as exp(β₁·x) with x up to 30, producing overflow values like −4.17e+182
and −2.34e+305 in the output CSVs.

The problem was most severe for high-inequality parameter combinations (β=0.5, c=0.5),
where exp(y) values span many orders of magnitude and the loss landscape is ill-conditioned.

## Fix

Three-part replacement in `estimate_indices_qrr` (both `mc_fld.py` and `mc_fld_no_r.py`):

1. **Convex solver.** Replace Nelder-Mead with sklearn `QuantileRegressor` (linear
   programming). The LP always converges to the global optimum and returns bounded betas.

2. **Geometric-mean normalised weights.** Use `wᵢ = exp(yᵢ − ȳ)` instead of raw
   `exp(yᵢ)`. This preserves the "higher outcome → higher weight" motivation of Y-scale
   QRR_linear (first-order Taylor: ρ_τ(Y−exp(a+bx)) ≈ exp(y)·ρ_τ(y−(a+bx))), but prevents
   weights from spanning many orders of magnitude in high-inequality scenarios.

3. **Isotonic correction.** Apply `IsotonicRegression` to β₀ and β₁ across quantile
   levels, same as IOQR. This eliminates residual quantile crossing that the weighted LP
   can still produce when weights are unequal.

## Resulting method identity

| Method | Weights | Isotonic |
|--------|---------|----------|
| BK     | equal   | no       |
| IOQR   | equal   | yes      |
| QRR_linear    | exp(y−ȳ) | yes    |

QRR_linear is thus outcome-weighted isotonic QR — a genuine three-way distinction from BK and
IOQR.

## Validation

After the fix, 50 MC replications on the worst-case scenario (α=0.5, β=0.5, c=0.5,
n=100, xmax=30) showed:

- BK:  min/max qZI = 0.573 / 0.997, n_negative = 0
- QRR_linear: min/max qZI = 0.222 / 1.000, n_negative = 0

No overflow, no negative values.
