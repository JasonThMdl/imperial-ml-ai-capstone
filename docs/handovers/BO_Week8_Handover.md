# BBO Capstone — Week 8 Handover

*Last updated: 2026-06-04 (entering Week 9)*

## Overview

Week 7 was the strongest round yet — **5 new bests** (F3, F4, F5, F7, F8), including the **F8 breakthrough** that broke a three-week plateau exactly as the calibrated trust-region GP predicted (9.7073 predicted vs 9.7083 actual). Week 8 strategies were derived and adversarially verified by an 8-function workflow (analyze → verify), then synthesised in `week8_bo.py` (reads `W7_data`). This document records the bests entering W9, the W7 scorecard, the W8 queries, and the conditional W9 plan.

---

## Current best results (entering Week 9, after W7 results)

| Fn | Dim | n | Best y | Best x |
|----|-----|---|--------|--------|
| F1 | 2 | 17 | 0.0000 | flat |
| F2 | 2 | 17 | 0.6136 | (0.718, 0.968) |
| F3 | 3 | 22 | −0.00689 | (0.466, 0.490, 0.426) |
| F4 | 4 | 37 | 0.65615 | (0.396, 0.418, 0.359, 0.414) |
| F5 | 4 | 27 | 7910.26 | (0.980, 0.995, 0.995, 0.990) |
| F6 | 5 | 27 | −0.2967 | (0.325, 0.349, 0.446, 0.770, 0.010) |
| F7 | 6 | 37 | 2.35497 | (0.005, 0.108, 0.515, 0.194, 0.359, 0.769) |
| F8 | 8 | 47 | 9.70830 | (0.034, 0.243, 0.098, 0.146, 0.348, 0.901, 0.169, 0.854) |

---

## Week 7 results — scorecard (best week so far)

| Fn | W6 best | W7 result | New best? | Verdict |
|----|---------|-----------|-----------|---------|
| F1 | 0.000 | ~0 | no | Flat; W7 input MIS-KEYED (0.993→0.900) — corner still unexplored |
| F2 | 0.6136 | 0.5310 | no | Tiny move, huge drop = NOISE confirmed; converged |
| F3 | −0.0077 | −0.00689 | **yes** | Manual x2↓ (0.517→0.49) worked |
| F4 | 0.650 | 0.65615 | **yes** | PI compounding |
| F5 | 7575 | 7910 | **yes** | Still rising past x1=0.98 (decelerating) |
| F6 | −0.2967 | −0.4963 | no | x2↓ hurt → both x4-up and x2-down now falsified |
| F7 | 2.2742 | 2.35497 | **yes** | Razor box; x3=0.515 beat 0.525 |
| F8 | 9.6220 | 9.70830 | **yes** | BREAKTHROUGH — trust-region GP, prediction nailed |

---

## Week 8 queries submitted

| Fn | Query | Method | Rationale |
|----|-------|--------|-----------|
| F1 | 0.994973-0.994844 | Maximin | Largest current gap (top-right). NB mis-keyed corner (0.9933,0.0057) still open |
| F2 | 0.717000-0.969100 | Manual centroid | Converged/noisy; verifier moved off the 0.717/0.965 spot that misfired |
| F3 | 0.465650-0.470000-0.425813 | Manual one-var | Continue the proven x2-down gradient (0.490→0.470) |
| F4 | 0.395840-0.417792-0.357908-0.411943 | GP PI ±0.018 | Calibrated (z=2.7); tighten around new best |
| F5 | 0.990000-0.994733-0.994727-0.990020 | Line search | Push x1→0.99 (still rising) |
| F6 | 0.325074-0.348782-0.453000-0.770325-0.010000 | Manual one-var | x3 0.446→0.453 (verifier tightened from 0.460); only clean untested lever |
| F7 | 0.005102-0.108260-0.503100-0.193856-0.358650-0.769170 | Manual one-var | Continue x3 down 0.515→0.503 (x3=0.515 beat 0.525) |
| F8 | 0.031623-0.192142-0.122970-0.095470-0.403959-0.862024-0.206800-0.808953 | Trust-region GP UCB | Tight box (W7 widths) around the breakthrough best; predicts ~9.77 |

**Workflow notes:** an 8-function analyze→verify workflow produced these. Adversarial verification caught two real issues — F2 about to re-sample the W7 misfire spot (→ cluster centroid), and F6's x3 step overshooting the good band (0.460→0.453). Overrides: F3 (agent's x3-up fought the negative correlation → kept the proven x2-down) and F8 (re-tightened the box after LOO flagged max|z|=3.6 on the wider version).

---

## Conditional Week 9 strategy

Load from **W8_data**. Adjust a copy of `week8_bo.py`.

- **F1** — continue maximin. **Re-submit the mis-keyed corner (0.993311, 0.005702) at some point** — it is still genuinely unexplored. If any query EVER returns > 0, switch to GP UCB exploiting it.
- **F2** — converged at ~0.6136 with noise std ~0.035. **Stop investing** — accept ≈0.614; the function is done. (If forced to submit, repeat the centroid.)
- **F3** — if x2=0.47 improved on −0.00689, keep stepping x2 down. If worse, x2≈0.49 is the optimum → then probe x3 down (follow corr −0.47) or accept the basin is mapped.
- **F4** — if W8 > 0.65615, tighten to ±0.012 around the new best. Keep GP PI (calibrated). Peak is sharp — do not widen.
- **F5** — if x1=0.99 still rising, push to 0.995 (the ceiling) for the final value. If it flattened, the plateau is reached → done.
- **F6** — if x3=0.453 improved on −0.2967, x3-up is the lever → continue. If worse, x3≈0.446 is right → probe x1 (the last clean untested lever) next. Surrogate stays untrusted (underfit).
- **F7** — if x3=0.503 > 2.355, x3 optimum is lower → continue down. If worse, the optimum is between 0.503 and 0.515 → bisect. Keep manual one-var (GP untrustworthy).
- **F8** — **the breakthrough lead.** If the tight trust-region point beat 9.7083, keep exploiting that neighbourhood with the same tight box. The GP is locally calibrated and predictive — trust its next UCB point but keep the box tight (wide boxes chase edges). Also worth a look: the alternate basin at 9.5985 (very different coords) may be a second optimum.

---

## Standing notes / pending

- LOO calibration check is built into the weekly script — keep it. F8's global LOO is now ~3.6 (inflated by distant low-y outliers); the *local* trust-region fit remains accurate, so judge F8 on local behaviour, not the global number.
- **Hardest remaining:** F6 (surrogate underfit; probing one lever at a time — only x1 left after x3) and F2 (converged). F8 and F4 are the active winners; F5 nearly maxed; F7 converging on x3.
- Pending docs (not queries): README method-mix update; `JUSTIFICATION.md` with citations (Turner et al. 2021 trust-region/ensembling; HEBO/Cowen-Rivers 2020 heteroscedastic+warped; Jones 1998 EI; Snoek 2012/2014; Srinivas 2010 UCB); post Module reflections.

---

## Progress tracker

| Fn | W5 | W6 | W7 | W8 query | Trend |
|----|----|----|----|----------|-------|
| F1 | 0.000 | 0.000 | 0.000 | Maximin | Flat |
| F2 | 0.613 | 0.6136 | 0.6136 | Centroid | Converged — stop |
| F3 | −0.0077 | −0.0077 | −0.00689 | x2↓ | Improving slowly |
| F4 | 0.566 | 0.650 | 0.65615 | PI ±0.018 | Strong, compounding |
| F5 | 6984 | 7575 | 7910 | x1→0.99 | Rising, decelerating |
| F6 | −0.297 | −0.297 | −0.297 | x3↑ | Stalled (probing levers) |
| F7 | 2.274 | 2.274 | 2.355 | x3↓ | Improving |
| F8 | 9.622 | 9.622 | 9.708 | Trust-region GP | BREAKTHROUGH — exploit |
