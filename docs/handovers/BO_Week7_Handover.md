# BBO Capstone — Week 7 Handover

*Last updated: 2026-06-04 (entering Week 8)*

## Overview

Week 7 folded in the two items deferred from the W6 handover — a standing **leave-one-out (LOO) calibration diagnostic** and an **input-warped GP** prototype for F3 — and acted on the Turner et al. (2021) BBO-Challenge analysis by moving F8 to a **trust-region local GP** (the approach that dominated that competition). Queries submitted via `week7_bo.py` (reads `W6_data`). This document records the bests entering W8, the W6 scorecard, the W7 queries, the diagnostic findings, and the conditional W8 strategy.

---

## Current best results (entering Week 8, after W6 results)

| Fn | Dim | n | Best y | Best x |
|----|-----|---|--------|--------|
| F1 | 2 | 16 | 0.0000 | flat — all queries ≈ 0 |
| F2 | 2 | 16 | 0.6136 | (0.718, 0.968) |
| F3 | 3 | 21 | −0.0077 | (0.466, 0.517, 0.426) |
| F4 | 4 | 36 | 0.6502 | (0.404, 0.410, 0.361, 0.426) |
| F5 | 4 | 26 | 7575.46 | (0.960, 0.995, 0.995, 0.990) |
| F6 | 5 | 26 | −0.2967 | (0.325, 0.349, 0.446, 0.770, 0.010) |
| F7 | 6 | 36 | 2.2742 | (0.005, 0.101, 0.525, 0.182, 0.365, 0.771) |
| F8 | 8 | 46 | 9.6220 | (0.027, 0.291, 0.080, 0.181, 0.290, 0.940, 0.126, 0.804) |

---

## Week 6 results — scorecard

| Fn | W5 best | W6 result | New best? | Verdict |
|----|---------|-----------|-----------|---------|
| F1 | 0.000 | ~0 | no | Flat (machine-zero) |
| F2 | 0.6131 | 0.6136 | **yes** | Converged at x1≈0.718 |
| F3 | −0.0077 | −0.0119 | no | Pushing x2/x3 up overshot |
| F4 | 0.566 | 0.650 | **yes** | Big win; PI compounding |
| F5 | 6984 | 7575 | **yes** | Still climbing (gain decelerating) |
| F6 | −0.297 | −0.462 | no | x4 0.77→0.80 hurt → x4 optimum ≈0.77 |
| F7 | 2.274 | 2.267 | no | x3=0.535 just shy of 0.525 peak |
| F8 | 9.622 | 9.599 | no | 3rd straight miss; ensemble unreliable |

Score: 3 wins (F2, F4, F5), 1 near-miss (F7), rest misses. The F6/F8 misses + the LOO findings motivated the W7 method changes.

---

## Week 7 queries submitted

| Fn | Query | Method | Rationale |
|----|-------|--------|-----------|
| F1 | 0.993311-0.005702 | Maximin | Bottom-right corner — last big unexplored gap |
| F2 | 0.717217-0.964989 | GP EI | Converged; confirm in a tiny box |
| F3 | 0.465650-0.490000-0.425813 | **Manual local** | Surrogate untrustworthy; best with x2 0.517→0.49 |
| F4 | 0.395790-0.418121-0.358689-0.414424 | GP PI ±0.015 | Calibrated (z=2.6); tighten around new best |
| F5 | 0.980000-0.994733-0.994727-0.990020 | Line search | Push x1→0.98; plateau test (poly pred ≈7722) |
| F6 | 0.325074-0.270000-0.445668-0.770325-0.010000 | Manual one-var | Revert x4 to 0.77; change ONLY x2 (0.349→0.27) |
| F7 | 0.005102-0.108260-0.515060-0.193856-0.358650-0.769170 | GP EI razor box | x3 pinned [0.515,0.532] around 0.525 |
| F8 | 0.034233-0.243235-0.098423-0.145451-0.347997-0.901040-0.169017-0.853815 | **Trust-region GP** | Local GP UCB; calibrated, predicts 9.71 > 9.622 |

---

## Diagnostics this week

**LOO calibration (now a standing check; flag |z|>3 = uncertainty not trustworthy):**

| Fn | std(z) | max\|z\| | Status |
|----|--------|----------|--------|
| F2 | 1.39 | 3.0 | borderline ✓ |
| F3 (unwarped) | 2.43 | 10.0 | ⚠ untrustworthy |
| F3 (warped) | 2.17 | 8.6 | ⚠ still untrustworthy |
| F4 | 0.78 | 2.6 | ✓ calibrated |
| F7 | 1.69 | 7.9 | ⚠ untrustworthy |
| F8 (trust-region) | 1.09 | 2.8 | ✓ calibrated |

**Key findings:**
- **F8 trust-region GP is the week's best move.** Calibrated *and* its UCB point predicts **9.71 > the stubborn 9.622** — the first credible shot at breaking F8's three-week plateau. Validates the TuRBO/local-model approach (Turner et al., 2021).
- **Input warping on F3 — tried and rejected.** Beta-CDF warping (Snoek et al., 2014), params fit by marginal-likelihood coordinate descent, reduced the cliff (max|z| 10.0 → 8.6) but did NOT make the GP trustworthy, and its UCB pointed back at x2≈0.54 — the same overshoot that failed in W6. Lesson: warping overfit (better LML, no better calibration); F3 is best handled with tight **local/manual** steps, not a global model.
- **F3 and F7 flagged as miscalibrated** → F3 overridden to a manual local step; F7 kept only because its box is tight enough to be effectively a trust region around the known best.

---

## Conditional Week 8 strategy

Load from **W7_data**. Adjust a copy of `week7_bo.py`.

- **F1** — continue maximin (next largest gap). If any query EVER returns > 0, switch immediately to GP UCB exploiting that region.
- **F2** — converged. If W7 ≤ 0.6136, accept ≈0.614 as the optimum and stop investing here.
- **F3** — if x2=0.49 improved on −0.0077, the x2 optimum is below 0.517 → keep stepping x2 down. If worse, the peak is at x2≈0.517 → hold and probe x3 instead. Keep F3 **local/manual** (surrogate untrustworthy).
- **F4** — if W7 > 0.650, tighten to ±0.01 around the new best. Keep GP PI.
- **F5** — if x1=0.98 ≈ 7722 (still rising), push x1→0.99. If it flattened or dropped, the plateau is reached → tight EI box around the best x1.
- **F6** — if x2=0.27 improved on −0.2967, x2-low is the lever → continue down. If worse, x2≈0.349 was right → probe x3 (raise toward 0.50) next, one variable at a time.
- **F7** — if W7 > 2.2742, tighten around the new point. Else x3≈0.525 confirmed → fine-tune x4/x5/x6 in a razor box.
- **F8** — **if the trust-region GP point beat 9.622, this is the breakthrough — exploit that neighbourhood hard.** If not, the local GP is well-calibrated, so trust its next suggestion; also consider the alternate basin (a 9.5985 point sits at very different coords — possible second optimum).

---

## Standing TO-DO / pending docs

- LOO calibration check is now built into `week7_bo.py` — keep it in every future script.
- **F6/F8 remain the hardest.** F8's trust-region GP is the promising new lead; F6 is being probed one variable at a time (surrogate underfit at n=26 in 5D).
- Pending (not queries): update README to current method mix; add `JUSTIFICATION.md` citing the literature (incl. **Turner et al. 2021** for trust-region + ensembling, **HEBO / Cowen-Rivers et al. 2020** for heteroscedastic+warped BO, Snoek 2014 warping); post the Module reflections.

---

## Progress tracker

| Fn | W4 | W5 | W6 | W7 query | Trend |
|----|----|----|----|----------|-------|
| F1 | 0.000 | 0.000 | 0.000 | Maximin | Flat |
| F2 | 0.613 | 0.613 | 0.6136 | Confirm | Converged |
| F3 | −0.018 | −0.0077 | −0.0077 | Manual x2↓ | Stalled (cliff) |
| F4 | 0.524 | 0.566 | 0.650 | PI ±0.015 | Strong, compounding |
| F5 | 5538 | 6984 | 7575 | x1→0.98 | Rising, decelerating |
| F6 | −0.357 | −0.297 | −0.297 | Manual x2↓ | Stalled |
| F7 | 2.274 | 2.274 | 2.274 | Razor box | Stalled near peak |
| F8 | 9.622 | 9.622 | 9.622 | Trust-region GP | Plateau — new lead |
