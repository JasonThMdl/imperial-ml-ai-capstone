# BBO Capstone — Week 4 Handover

## Overview

Week 4 introduced a neural network surrogate (bootstrap MLP ensemble) for the higher-dimensional functions (F6–F8), replacing the GP there. For F2–F5 the GP was retained where it remained reliable. Week 4 queries have been submitted; results will arrive as W4_data. This document records the current best results, the W4 queries submitted, key lessons learned, and the recommended Week 5 strategy for each function.

---

## Current best results (entering Week 5)

| Fn | Dim | n (W3) | Best y | Best x |
|----|-----|--------|--------|--------|
| F1 | 2 | 13 | 0.0000 | (0.731, 0.733) — all queries zero |
| F2 | 2 | 13 | 0.6131 | (0.721, 0.970) |
| F3 | 3 | 18 | −0.0287 | (0.470, 0.483, 0.401) |
| F4 | 4 | 33 | 0.5235 | (0.408, 0.411, 0.347, 0.433) |
| F5 | 4 | 23 | 4486.13 | (0.544, 0.995, 0.995, 0.990) |
| F6 | 5 | 23 | −0.3572 | (0.361, 0.338, 0.456, 0.746, 0.056) |
| F7 | 6 | 33 | 2.2742 | (0.005, 0.101, 0.525, 0.182, 0.365, 0.771) |
| F8 | 8 | 43 | 9.6220 | (0.027, 0.291, 0.080, 0.181, 0.290, 0.940, 0.126, 0.804) |

---

## Week 4 queries submitted

| Fn | Query submitted | Surrogate | Rationale |
|----|----------------|-----------|-----------|
| F1 | 0.419725 − 0.463133 | Maximin | Space-filling; all previous returns zero |
| F2 | 0.714350 − 0.969777 | GP EI | Tight exploit: x1≈0.71, x2≈0.97 |
| F3 | 0.462875 − 0.451708 − 0.375416 | GP UCB κ=2.0 | Near new best; x3 capped at 0.44 |
| F4 | 0.402554 − 0.384562 − 0.381604 − 0.414273 | GP EI | ±0.07 tight box around only positive point |
| F5 | 0.778636 − 0.994733 − 0.994727 − 0.990020 | GP EI | x1 pushed to 0.779; x2/x3/x4 at ceiling |
| F6 | 0.220000 − 0.318954 − 0.380000 − 0.850000 − 0.005000 | NN UCB κ=2.0 | x5 pinned near 0; x4 high |
| F7 | 0.005000 − 0.070000 − 0.650000 − 0.130000 − 0.342616 − 0.880000 | NN UCB κ=2.0 | x3 corrected to 0.65 after W3 error |
| F8 | 0.005000 − 0.350000 − 0.005000 − 0.250000 − 0.400000 − 0.990000 − 0.050000 − 0.960000 | NN UCB κ=2.0 | x1/x3 pushed to minimum; x6/x8 at ceiling |

---

## Key lessons from Week 3 → Week 4

**F1** — After 13 queries all returning 0, the function appears genuinely flat in [0,1]². No GP or NN can extract signal that does not exist. Continue space-filling to be thorough but do not invest strategy effort here.

**F2** — The Week 3 query dropped x2 from 0.970 to 0.867 and output fell from 0.613 to 0.549. x2 ≥ 0.945 is a hard constraint. x1 around 0.71–0.72 is optimal. Do not explore outside these bounds.

**F3** — Week 3 hit a new best (−0.0287) at x3 = 0.401. Previous best was at x3 = 0.340. The x3 correlation is −0.55 overall but the best two points both have x3 in [0.34, 0.41] — do not extrapolate to x3 > 0.44.

**F4** — Only 1 point with y > 0 in 33 queries. Week 3's UCB query deviated slightly and returned −2.59. This function has a very narrow feasible region around (0.41, 0.41, 0.35, 0.43). The NN ensemble was unreliable here (sigma = 6.6); GP EI in a tight box is safer. Do not use wide exploration.

**F5** — Clear upward trajectory as x1 increases: x1 = 0.265 → 4257, x1 = 0.304 → 4270, x1 = 0.544 → 4486. x2/x3/x4 must stay near 0.995. W4 pushes x1 to 0.779; GP predicts ≈4857. If that comes back significantly lower, the plateau has been reached.

**F6** — x5 is the dominant variable (r = −0.66): it must stay near 0. x4 must be high (r = +0.57). Week 3 query raised x5 from 0.056 to 0.006 while dropping x3 from 0.456 to 0.275 — result dropped to −0.683. W4 pins x5 ≤ 0.005 and keeps x4 ≥ 0.85.

**F7** — Week 3 dropped x3 from 0.525 to 0.272 and output fell from 2.274 to 1.373. x3 ≈ 0.50 is critical. W4 corrects this: x3 = 0.650 is the upper bound of the allowed range. x1 must remain near 0 (r = −0.53).

**F8** — Progressive improvement across all three weeks (9.599 → 9.622). Strong signals: x1 and x3 both r = −0.69 (must be very low); x7 r = −0.43 (keep low). W4 pushes x1 = x3 = 0.005, x6 = 0.990, x8 = 0.960.

---

## Week 5 strategy (conditional on W4 results)

Load from **W4_data**. Update the script template from `week4_bo.py`, change `BASE` to point at `W4_data`, and adjust per-function strategies as follows.

### F1 — 2D Radiation
**If W4 = 0 again**: Accept the function is flat. Submit one final space-filling point using maximin. No further strategy investment needed.
**If W4 > 0**: Immediately switch to GP UCB exploiting that region — treat it as a strong signal.

### F2 — 2D Noisy ML
**If W4 > 0.613**: Exploit further; tighten the box to x1 ∈ [0.710, 0.720], x2 ∈ [0.965, 0.975].
**If W4 ≤ 0.613**: The best may be at (0.721, 0.970). Try x1 ∈ [0.715, 0.730], x2 ∈ [0.965, 0.975] with EI.
Hard constraint: x2 ≥ 0.945 always.

### F3 — 3D Non-smooth Drug
**If W4 > −0.0287**: Exploit around the new best; keep x3 ∈ [0.33, 0.44].
**If W4 ≤ −0.0287**: The function may be near its best. Try a slightly different x2 value while keeping x1 ≈ 0.47 and x3 ≈ 0.38.
Do not push x3 above 0.44.

### F4 — 4D Multimodal Warehouse
**If W4 > 0.524**: Exploit with an even tighter box (±0.04) around the new best.
**If W4 ≤ 0**: The feasible region may be smaller than ±0.07. Try the exact best point ± 0.03 and accept that improvement may be marginal.
Use GP EI only — do not use NN here.

### F5 — 4D Chemical Process
**If W4 ≈ 4857 (GP prediction)**: Push x1 further — try x1 ∈ [0.78, 0.90]. If GP prediction is confirmed, the function is still rising.
**If W4 < 4486 (regression)**: The optimum is near x1 = 0.54. Switch to a tight exploit box around (0.544, 0.995, 0.995, 0.990) with EI.
**If W4 ∈ [4486, 4600]**: Plateau forming. Fine-tune x1 in [0.55, 0.70] with EI.

### F6 — 5D Cake
**If W4 > −0.357**: Exploit the new best; keep x5 ≤ 0.005, x4 ≥ 0.80.
**If W4 ≤ −0.357**: Still stuck. Widen x3 slightly (try 0.42–0.58) while keeping x5 ≤ 0.01 and x4 ∈ [0.72, 0.88]. NN UCB κ=1.8.

### F7 — 6D ML Hyperparameter
**If W4 > 2.274**: Exploit around the new best; note the x3 value that gave improvement and tighten there.
**If W4 ≤ 2.274**: The function peaks at x3 ≈ 0.525. Return to a tight box around the known best (0.005, 0.101, 0.525, 0.182, 0.365, 0.771) with EI. Do not lower x3 below 0.45.

### F8 — 8D
**Always**: Keep x1 < 0.06, x3 < 0.06, x6 > 0.88, x8 > 0.80, x7 < 0.15.
**If W4 > 9.622**: Tighten x1 and x3 further (try < 0.01) and raise x6 toward 0.99.
**If W4 ≤ 9.622**: Try varying x2 (currently 0.35 — test lower at 0.05–0.15) and x5 (try 0.25–0.35) while holding all other constraints fixed.

---

## Script setup for Week 5

```python
# Change this line in week4_bo.py (save as week5_bo.py)
BASE = os.path.join("..", "..", "data", "W4_data")  # repo-relative
```

Adjust per-function bounds and acquisition functions based on the W4 results as described above. The `nn_best()` and `fit_gp()` helper functions can be reused unchanged.

---

## Progress tracker

| Fn | W1 best | W2 best | W3 best | W4 query | Trend |
|----|---------|---------|---------|----------|-------|
| F1 | 0.000 | 0.000 | 0.000 | — | Flat |
| F2 | 0.613 | 0.613 | 0.613 | Exploit | Stable |
| F3 | −0.035 | −0.035 | −0.029 | Exploit | Improving slowly |
| F4 | 0.524 | 0.524 | 0.524 | Exploit | Stalled — fragile |
| F5 | 4257 | 4270 | 4486 | Push x1 | Strong uptrend |
| F6 | — | −0.357 | −0.357 | Corrected | Stalled |
| F7 | — | 2.274 | 2.274 | Corrected x3 | Stalled — W3 error |
| F8 | 9.570 | 9.599 | 9.622 | Tighter constraints | Steady improvement |
