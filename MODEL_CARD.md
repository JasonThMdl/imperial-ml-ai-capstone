# Model Card — Adaptive Bayesian Optimisation Pipeline

## Model Description

An adaptive, human-in-the-loop Bayesian optimisation pipeline that proposes one query per week for each of eight unknown functions.

**Input:** all observed `(x, y)` pairs for a function so far. `x` is an `(n, d)` array in `[0, 1]^d` (d = 2–8, n = 10–51) and `y` is an `(n,)` array of real-valued outputs.

**Output:** one proposed query `x_next ∈ [0, 1]^d` per function, formatted for the portal (`0.123456-0.654321-…`), plus the surrogate's predicted mean and standard deviation at that point.

**Model Architecture:**

1. **Surrogate.** Gaussian Process regression (scikit-learn `GaussianProcessRegressor`, `normalize_y=True`).
   - Kernel: `Constant × Matérn(ARD, ν ∈ {½, 3/2, 5/2}) [+ WhiteKernel]`, with ν chosen per function.
   - Hyperparameters fitted by marginal likelihood (L-BFGS-B, 10 restarts).
2. **Acquisition.** UCB (`μ + κσ`, κ 1.5–3.5), EI or PI, selected per function.
3. **Search space.** Either the full box or a **trust region** around the current best, with per-dimension half-widths of 0.01–0.06. Maximised by dense random scan (≈ 10⁵ candidates).
4. **Strategy layer.** Per-function rules chosen from evidence:
   - Line search (F5) and maximin space-filling (F1).
   - Manual one-variable steps when the GP fails calibration.
   - Tried and retired: a bootstrap MLP ensemble (15 × 2-layer, 64 ReLU) and Differential Evolution.
5. **Diagnostics.** A leave-one-out standardised-residual check runs each week (|z| > 3 means the uncertainty is not trusted), plus an out-of-sample back-test of predictions.

The weekly decisions were made with AI assistance (Claude) for analysis and code. Every query was reviewed and submitted by me, and the reasoning is logged in `docs/handovers/`.

## Performance

Performance is measured two ways.

**(a) Optimisation outcome:** best observed output after 11 weekly queries vs the best in the initial sample.

| Fn | Dim | Initial best | Final best |
|----|-----|--------------|------------|
| F1 | 2 | ≈ 0 | ≈ 0 |
| F2 | 2 | 0.611 | 0.637 |
| F3 | 3 | −0.0348 | −0.00689 |
| F4 | 4 | −4.03 | 0.656 |
| F5 | 4 | 1,089 | 8,585 |
| F6 | 5 | −0.714 | −0.297 |
| F7 | 6 | 1.365 | 2.763 |
| F8 | 8 | 9.598 | 9.908 |

Seven of eight functions improved. F1 showed no detectable signal.

![Best observed per week](figures/best_so_far.png)

**(b) Surrogate accuracy:** for each weekly query in weeks 4–11, the GP was refit on only the data available beforehand and its prediction compared with the actual output (`notebooks/bbo_capstone.ipynb`, §8).

| Fn | MAE | Within ±2σ | Spearman ρ |
|----|-----|------------|------------|
| F4 | 0.151 | 8/8 | 0.64 |
| F7 | 0.084 | 6/8 | 0.86 |
| F8 | 0.024 | 7/8 | 0.90 |

LOO calibration on the final data: F2, F4 and F8 are calibrated (max|z| ≤ 2.7). F3 (17.8), F5 (6.8), F6 (3.8) and F7 (4.4) are not, which is why those functions were handled locally or manually.

![Back-test](figures/backtest_calibration.png)

## Limitations

- **Small data.** 21–51 points in 2–8D. Global surrogates are unreliable, and "best found" cannot be certified as the global optimum, especially for F6–F8.
- **Stationary kernels.** A single length-scale per dimension cannot represent cliffs (F3) or regions that change sharply. Input warping did not fix this.
- **Homoscedastic noise.** One global noise term is assumed. F2's noise appears to depend on the output level.
- **Human-in-the-loop.** Strategy switches and manual steps add judgement that is documented but not fully automated. Another person might make different calls.
- **Path dependence.** Results depend on the specific sequence of 11 queries. The pipeline cannot be "re-run" against the hidden functions.
- **Synthetic setting.** No real units, constraints or costs, so it says nothing about safety-critical use.

## Trade-offs

- **Local vs global.** Trust regions gave reliable, steady gains (F8, F7) but can lock onto one basin. A better basin elsewhere (possible for F6, F7) would be missed. A planned exploration week (W10) partly mitigated this.
- **PI vs EI/UCB.** PI found F4's narrow peak and stayed there, but it is too timid to discover new regions.
- **Surrogate vs no surrogate.** Line search beat the GP on monotone F5 but would fail on any multimodal surface.
- **NN ensemble vs GP.** The ensemble is more flexible and scales better, but at n < 50 its uncertainty was miscalibrated (it could not reproduce its own best training point), so the GP won.
- **Noise.** On F2, chasing apparent gains smaller than the noise (σ ≈ 0.03) wastes queries. The pipeline deliberately stopped optimising there.
