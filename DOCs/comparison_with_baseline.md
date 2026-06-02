# Comparison with baselines — misspecification study (n=50, 1000 MC reps)

## Methods

### Proposed methods
- **IOQR** — Isotonic Ordinary Quantile Regression. Fits standard linear QR on log(Y) at 99
  quantile levels, then applies isotonic regression to the estimated β₀(τ) and β₁(τ) curves
  to enforce monotonicity. The core contribution of the paper.
- **IAQR** — Isotonic Approximate Quantile Regression. Replaces the non-smooth check function
  with a smooth approximation (average of two differentiable surrogates), uses the estimated
  IQR at the mean of x as bandwidth, then applies isotonic regression. Designed for better
  small-sample behaviour.

### Literature competitors
- **BK** (Bassett & Koenker 1982) — standard unconstrained linear QR on log(Y) at 99 levels.
  No monotonicity enforcement; can produce crossing quantile functions.
- **BRW** (Bondell, Reich & Wang 2010) — simultaneous noncrossing QR via convex programming
  (cvxpy/ECOS). Enforces Q(τ₁|x) ≤ Q(τ₂|x) for all observed x jointly.
- **WL1** (Wu & Liu 2009) — stepwise noncrossing QR. Fits the median first, then adds
  quantile levels one at a time upward and downward using a noncrossing constraint from the
  previous level.
- **QRR_linear** — Outcome-weighted isotonic QR on log(Y) with weights wᵢ = exp(yᵢ − ȳ).
  Inspired by Farcomeni & Geraci (2024) QRR but not a faithful implementation (their method
  uses a linearized iterative algorithm on quantile ratios directly; ours is a weighted LP on
  the log-Y quantile function). Added as a QRR-flavored benchmark at the request of reviewers
  #4 and #5.

### Simple baselines (added for reviewer #3)
- **Strat** — Stratified empirical estimator. Bins x into K=5 equal-width groups, computes
  the empirical quantile function of Y within each bin, and integrates to obtain qZI/qDI.
  The simplest thing a practitioner would try.
- **IQRR** — Inter-Quantile Ratio Regression. Fits QR at only two quantile levels {0.25, 0.75}
  and linearly interpolates the estimated β curves to the full 99-level grid. Represents the
  "just use the IQR" approach.

---

## Misspecification scenarios

Three DGPs where the log-linear model assumed by all QR-based methods is violated:

- **3a** — Quadratic CQF: log(Q_Y(p|x)) = α + β·(logit(p) + c·p·x + **c₂·p·x²**).
  Breaks the linearity-in-x assumption.
- **3b** — Log-normal heteroscedastic: log(Y)|X=x ~ Normal(μ₀, (σ₀+σ₁x)²).
  Non-EFLD marginal shape; β₁(p) is monotone so the log-linear structure is mild.
- **3c** — Non-monotone β₁: log(Q_Y(p|x)) = logit(p) + c_bell·4·p·(1−p)·x.
  β₁(p) is bell-shaped (maximum at p=0.5, zero at p=0 and 1). Directly tests
  whether the isotonic correction on β₁ is harmful when the truth is non-monotone.

---

## Findings

### Against the simple baselines (Strat, IQRR)

**IOQR and IAQR beat both baselines in every scenario at every x value.**
Strat has MSE 3–6× higher than IOQR in 3a and 3b, and remains poor throughout 3c.
IQRR is 2–4× worse than IOQR at small x in 3a and at large x in 3b and 3c.
The full QR machinery is justified: using more quantile levels and proper smoothing
consistently outperforms crude binning or a two-level approximation, even under
misspecification.

### Against the literature competitors (BK, BRW, WL1)

**3a (quadratic):** IOQR and IAQR are the best methods, with BRW and WL1 close behind.
BK is slightly worse at small x. All five QR-based methods substantially outperform
the baselines.

**3b (log-normal):** All five QR-based methods perform nearly identically — MSEs differ
by less than 20% at any x. The log-normal misspecification is mild enough that the
log-linear model provides an adequate approximation for all methods.

**3c (non-monotone β₁):** The results are split by x range. At moderate x (5–20),
BK and WL1 have lower MSE than IOQR and IAQR — the isotonic correction introduces
bias when the true β₁ is non-monotone. At large x (x=30), this reverses: BK's MSE
for qZI reaches 8.76×10⁻³ versus IOQR's 1.91×10⁻³, because without isotonic
correction the noisy extreme-quantile estimates of β₁ dominate the qZI integral.
This is the empirical manifestation of the bias–variance trade-off of isotonic
regression that reviewer #9 asked about.

### Against QRR_linear

**IOQR and IAQR clearly dominate QRR_linear in all three scenarios.**

| DGP | x | IOQR qZI MSE×10³ | QRR_linear qZI MSE×10³ | ratio |
|-----|---|------------------|-----------------|-------|
| 3a  | 1 | 9.63             | 40.12           | 4.2×  |
| 3a  | 30| 2.19             | 25.12           | 11.5× |
| 3b  | 1 | 5.62             | 7.90            | 1.4×  |
| 3b  | 30| 5.00             | 8.73            | 1.7×  |
| 3c  | 1 | 5.36             | 82.66           | 15.4× |
| 3c  | 10| 1.48             | 37.07           | 25.0× |

This is a stronger result than the well-specified case, where QRR_linear ≈ BK ≈ IOQR.
Under misspecification, QRR_linear's outcome-level weighting concentrates influence on
high-outcome observations, which are not representative of the full conditional
distribution. This distorts the estimated β curves across the entire quantile grid.
Reviewers #4 and #5 argued that a QRR benchmark is natural because it "directly
targets quantile ratios" — but empirically, IOQR and IAQR are more robust under
all three misspecification types tested here.

---

## Summary

| Scenario | Best overall | IOQR vs baselines | IOQR vs BK/BRW/WL1 | IOQR vs QRR_linear |
|---|---|---|---|---|
| 3a quadratic | IOQR / IAQR | wins clearly | slight advantage | wins 4–12× |
| 3b log-normal | all QR-based tie | wins clearly | no meaningful diff | wins 1.4–1.7× |
| 3c non-mono β₁ | mixed by x | wins clearly | loses at mod. x, wins at large x | wins 5–25× |

The simple baselines are never competitive. QRR_linear is never competitive under
misspecification. The main open question is the bias–variance trade-off with BK/WL1
in scenario 3c, which is an honest limitation that should be discussed in the paper.
