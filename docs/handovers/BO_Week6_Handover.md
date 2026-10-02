# BBO Capstone — Week 6 Handover

*Last updated: 2026-06-04 (entering Week 7)*

## Overview

Week 6 queries have been submitted (`week6_bo.py`, reads from `W5_data`). The portfolio of methods is now genuinely mixed: GP for the low-dimensional smooth functions, a manual/data-driven rule where the surrogate proved unreliable, a 1D line search for the monotone F5, and a bootstrap MLP ensemble for the high-dimensional functions. This document records current bests (after W5 results), the W6 queries submitted, the model-diagnostic findings on non-stationarity/heteroscedasticity, and the agreed to-do list for Week 7.

---

## Current best results (entering Week 7, after W5 results)

| Fn | Dim | n | Best y | Best x |
|----|-----|---|--------|--------|
| F1 | 2 | 15 | 0.0000 | flat — all queries ≈ 0 (one −0.006) |
| F2 | 2 | 15 | 0.6131 | (0.721, 0.970) |
| F3 | 3 | 20 | −0.0077 | (0.466, 0.517, 0.426) |
| F4 | 4 | 35 | 0.5661 | (0.405, 0.401, 0.369, 0.423) |
| F5 | 4 | 25 | 6983.62 | (0.920, 0.995, 0.995, 0.990) |
| F6 | 5 | 25 | −0.2967 | (0.325, 0.349, 0.446, 0.770, 0.010) |
| F7 | 6 | 35 | 2.2742 | (0.005, 0.101, 0.525, 0.182, 0.365, 0.771) |
| F8 | 8 | 45 | 9.6220 | (0.027, 0.291, 0.080, 0.181, 0.290, 0.940, 0.126, 0.804) |

---

## Week 6 queries submitted

| Fn | Query | Method | Rationale |
|----|-------|--------|-----------|
| F1 | 0.224981-0.568320 | Manual maximin | Pre-planned point "W13"; fills sparse left-central region |
| F2 | 0.718000-0.968264 | GP EI | x1=0.715→0.598 still below best; nudge toward proven x1=0.721 |
| F3 | 0.496377-0.543179-0.416652 | GP UCB κ=1.5 | Raising x2/x3 drove the W5 best; allow slight push higher |
| F4 | 0.404076-0.409973-0.361108-0.425651 | GP PI ±0.025 | PI delivered new best 0.566; tighten the exploit |
| F5 | 0.960000-0.994733-0.994727-0.990020 | 1D line search | Trend *accelerating* (5538→6984); push x1 to 0.96 |
| F6 | 0.325074-0.348782-0.445668-0.800000-0.010000 | Manual x4 step | Ensemble underfit (see below); move only x4 (0.770→0.80) |
| F7 | 0.005102-0.099851-0.534957-0.188072-0.359369-0.781119 | GP EI | x3=0.541→2.244 just missed; razor-tight around x3=0.525 |
| F8 | 0.013200-0.329932-0.054682-0.165517-0.334196-0.964358-0.144260-0.843960 | NN UCB κ=1.5 | Two explores regressed; consolidate around W3 best (9.622) |

**F6 note:** switched off the NN ensemble this week. It predicted −0.64 at the known best whose true value is −0.30 — i.e. it could not reproduce a training point (25 pts in 5D, bootstrap drops the rare good sample), so UCB was just chasing variance to box corners. Replaced with a clean one-variable step (raise x4 only) so next week's result is cleanly attributable. Prior high-x4 points (0.84–0.85) failed *only* because their x3 was wrong — "high x4 with correct x3" is genuinely untested.

---

## F1 — 10-week maximin plan (pre-computed)

F1 returns ≈0 everywhere (13 zeros, 2 tiny negatives in 15 queries), but with 10 queries left in a 2D space we keep searching rather than accept 0. All remaining points are pre-planned by maximin (each fills the largest gap). Submit one per week:

| Week | F1 query | | Week | F1 query |
|------|----------|-|------|----------|
| W6 ✅ | 0.224981-0.568320 (point "W13") | | W11 | 0.249182-0.994785 |
| W7 | 0.682808-0.006320 | | W12 | 0.620859-0.218897 |
| W8 | 0.993324-0.006131 | | W13 | submitted in W6 |
| W9 | 0.012830-0.694620 | | W14 | 0.126726-0.192301 |
| W10 | 0.994330-0.993300 | | W15 | 0.994890-0.768237 |

(W6 used the "W13" point out of order by choice; remaining order can be re-optimised if desired.) By W15 no point in [0,1]² will be >~0.21 from an observation — if a positive peak exists we will have come within 0.21 of it.

---

## Model diagnostics — non-stationarity & heteroscedasticity

**Why:** Every GP so far has used a **stationary** kernel (Matérn/RBF — covariance depends only on distance, one global length-scale) and a **homoscedastic** noise term (single global `WhiteKernel`). These assumptions were never tested. Ran a leave-one-out (LOO) standardized-residual check on `W5_data`: refit on n−1 points, predict the held-out point, compute `z = (y − μ)/σ`. A well-specified stationary + homoscedastic GP gives `z ~ N(0,1)` (mean≈0, std≈1, no large outliers).

| Fn | std(z) | max\|z\| | corr(\|err\|, y) | Reading |
|----|--------|----------|------------------|---------|
| F2 | 1.26 | 2.9 | −0.61 | **Heteroscedastic** — error scales with output level; homoscedastic WhiteKernel mis-specified |
| F3 | 2.44 | **9.8** | −0.78 | **Non-stationary + heteroscedastic** — one point ~10σ off; a cliff the global length-scale cannot see |
| F4 | 0.80 | 2.7 | −0.35 | Calibrated *by luck* — large noise term (0.1) + tight box mask it, not correct modelling |
| F7 | 1.19 | 2.6 | −0.04 | Mild non-stationarity — a *cluster* of ~2.5 residuals = consistent overconfidence |

**Findings**
- **F3 is the worst case**: z=9.8 is a localized blow-up (GP confident right where the function jumped) — textbook non-stationarity + heteroscedastic noise.
- **F2 noise is not constant** — larger where y is low; a single global noise variance is wrong.
- **F4 only looks fine by luck** — the fat noise term inflates σ and the tight box keeps us local.
- **Key insight:** the tight local-box exploitation we already do (F4, F7) *is* the correct defence against non-stationarity at n=15–45 — you cannot fit spatially-varying length-scales or input-dependent noise from so few points (the TuRBO argument). We've been implicitly robust without naming it.

---

## TO-DO for Week 7 (agreed)

1. **Add the LOO calibration check to the weekly script** as a standing diagnostic. Flag any `|z| > 3` as "do not trust this GP's uncertainty here."
2. **F3** — prototype an **input-warped GP** (Snoek, Swersky, Zemel & Adams, 2014 — Beta-CDF input warping) to tame the z=9.8; otherwise keep treating F3 purely locally.
3. **F2** — accept local homoscedastic noise (cannot fit input-dependent noise at n=15; cf. Kersting et al., 2007). Mis-specification lives outside the high-y region we exploit.
4. For genuinely heteroscedastic uncertainty, lean on the **bootstrap ensemble (F6/F8)** as more points accumulate — its spread is naturally input-dependent once calibrated.
5. **Pending docs** (separate from queries): update README to current method mix; add `JUSTIFICATION.md` with the literature citations; post the Module-17 reflection.

*Diagnostic reproducer:* LOO standardized-residual script over `W5_data`, kernels per function as in `week6_bo.py`.

---

## Week 7 strategy (conditional on W6 results)

Load from **W6_data** once results arrive. Update `BASE` in a copy of `week6_bo.py`. Per-function expectations:

- **F1** — submit next maximin point (W7 = 0.682808-0.006320). If any query ever returns > 0, switch immediately to GP UCB exploiting that region.
- **F2** — if W6 > 0.613, tighten around the new point; else stay in x1∈[0.715,0.725], x2≥0.965.
- **F3** — if W6 > −0.0077, the upward x2/x3 push is working; continue. Else exploit tightly around (0.466, 0.517, 0.426).
- **F4** — if W6 > 0.566, tighten ±0.02 around the new best. Keep PI, not EI.
- **F5** — if still rising at x1=0.96, push x1 to 0.98. If it drops, the plateau is reached → tight box around the best x1.
- **F6** — if the x4=0.80 step improved on −0.297, push x4 to 0.85 (now with correct x3). If worse, x4≈0.77 was the local peak; revert and probe x1/x3 instead.
- **F7** — if W6 > 2.274, tighten around it. Else accept x3≈0.525 is the peak and fine-tune x4/x5/x6.
- **F8** — if W6 > 9.622, exploit the new point. Else 9.622 (W3) is robust; vary one untested dimension at a time.
