# BBO Capstone — Week 9 Handover

*Last updated: 2026-06-04 (entering Week 10)*

## Overview

Week 8 produced 4 new bests (F2-by-noise, F5, F7, F8). F8's trust-region GP continues to deliver (9.622 → 9.708 → 9.772, prediction matching actual two weeks running). Several functions are now converged or near-converged; the strategy has shifted to **allocating scarce queries by marginal return** — push the ones still rising (F5, F8), re-converge F4, settle the last levers on F3/F6/F7, and stop investing in F2. Queries built in `week9_bo.py` (reads `W8_data`).

---

## Current best results (entering Week 10, after W8 results)

| Fn | Dim | n | Best y | Best x |
|----|-----|---|--------|--------|
| F1 | 2 | 18 | 0.0000 | flat |
| F2 | 2 | 18 | 0.6368 | (0.717, 0.969) — noise-driven, see note |
| F3 | 3 | 23 | −0.00689 | (0.466, 0.490, 0.426) |
| F4 | 4 | 38 | 0.65615 | (0.396, 0.418, 0.359, 0.414) |
| F5 | 4 | 28 | 8088.30 | (0.990, 0.995, 0.995, 0.990) |
| F6 | 5 | 28 | −0.2967 | (0.325, 0.349, 0.446, 0.770, 0.010) |
| F7 | 6 | 38 | 2.35702 | (0.005, 0.108, 0.503, 0.194, 0.359, 0.769) |
| F8 | 8 | 48 | 9.77181 | (0.032, 0.192, 0.123, 0.095, 0.404, 0.862, 0.207, 0.809) |

---

## Week 8 results — scorecard

| Fn | W7 best | W8 result | New best? | Verdict |
|----|---------|-----------|-----------|---------|
| F1 | 0.000 | ~0 | no | Flat confirmed (top-right corner also 0) |
| F2 | 0.6136 | 0.6368 | **yes*** | *Noise draw, not real signal (σ≈0.14) — converged |
| F3 | −0.00689 | −0.0193 | no | x2 down to 0.47 overshot → x2 optimum ≈0.49 (bracketed) |
| F4 | 0.65615 | 0.6448 | no | Sharp peak; PI step landed slightly off |
| F5 | 7910 | 8088 | **yes** | Still rising at x1=0.99 (+178) |
| F6 | −0.2967 | −0.4157 | no | x3-up hurt → all three probed levers now falsified |
| F7 | 2.35497 | 2.35702 | **yes** | x3 down to 0.503 edged up |
| F8 | 9.70830 | 9.77181 | **yes** | Trust-region GP delivers again (predicted ≈ actual) |

---

## Week 9 queries submitted

| Fn | Query | Method | Rationale |
|----|-------|--------|-----------|
| F1 | 0.006728-0.722944 | Maximin | Largest current gap (left edge; corners now filled) |
| F2 | 0.717000-0.969100 | Re-confirm best | CONVERGED & noisy; no further investment |
| F3 | 0.465650-0.490000-0.400000 | Manual one-var | Probe last lever x3↓ 0.426→0.400 (corr −0.47) |
| F4 | 0.394258-0.418230-0.357700-0.413872 | GP PI ±0.010 | Re-converge on the sharp peak |
| F5 | 0.995000-0.994733-0.994727-0.990020 | Line search | Push x1 to the ceiling (final value) |
| F6 | 0.280000-0.348782-0.445668-0.770325-0.010000 | Manual one-var | Last clean lever x1↓ 0.325→0.280 |
| F7 | 0.005102-0.108260-0.490000-0.193856-0.358650-0.769170 | Manual one-var | Continue x3↓ 0.503→0.490 (proven direction) |
| F8 | 0.041516-0.160574-0.145419-0.120185-0.463014-0.822215-0.164524-0.834183 | Trust-region GP UCB | Tight box; calibrated (LOO z=2.1); predicts ≈9.83 |

---

## Conditional Week 10 strategy

Load from **W9_data**. Adjust a copy of `week9_bo.py`.

- **F1** — continue maximin. The function is flat (18 obs ≈ 0); accept best=0. Keep filling gaps only to be thorough; no strategy investment.
- **F2** — **STOP.** Converged at ~0.6136 real (0.6368 is noise). Submit the best point if a query is required, but expect no real gain.
- **F3** — if x3=0.40 improved on −0.00689, x3↓ is a lever → continue. If worse, the basin is fully mapped (x2≈0.49, x3≈0.426) → accept ≈−0.0069 and stop.
- **F4** — if W9 ≥ 0.65615, hold the ±0.010 PI box. If it keeps landing just below, the peak is essentially found (0.656) — accept it; the surface is too sharp to squeeze further reliably.
- **F5** — x1=0.995 is the ceiling. After this, x1 is exhausted; if still rising the gain is tiny — **F5 is essentially done** at its maximum x1.
- **F6** — if x1=0.28 improved on −0.2967, keep going down x1. If worse (likely — it is a weak lever), **all clean levers are exhausted → accept −0.2967** as the local optimum. F6 is the hardest function; do not over-spend.
- **F7** — if x3=0.49 > 2.357, continue x3 down (optimum still lower). If worse, optimum is in [0.49, 0.503] → bisect, or accept ≈2.357.
- **F8** — **the active winner.** Keep the tight trust-region GP UCB centred on the latest best. It has improved every week since the switch; trust its next calibrated point but keep the box tight (wide boxes chase edges). Watch for it to plateau like F5 eventually.

---

## Standing notes

- **Endgame framing:** with ~5 weeks left, most functions are converged or near it. Active gains remain mainly in **F8** (still climbing) and marginally **F4/F7**. F5 is at its ceiling; F1/F2 done; F3 basin mapped; F6 at its local optimum.
- LOO calibration is built into the script. F8's local fit stays calibrated (z=2.1); trust it.
- Pending docs (not queries): README method-mix update; `JUSTIFICATION.md` with citations (Turner 2021 trust-region/ensembling; HEBO/Cowen-Rivers 2020; Jones 1998 EI; Srinivas 2010 UCB; Snoek 2012/2014); post Module reflections.

---

## Progress tracker

| Fn | W6 | W7 | W8 | W9 query | Status |
|----|----|----|----|----------|--------|
| F1 | 0.000 | 0.000 | 0.000 | Maximin | Flat — done |
| F2 | 0.6136 | 0.6136 | 0.6368* | Re-confirm | Converged (noise) — stop |
| F3 | −0.0077 | −0.00689 | −0.00689 | x3↓ | Basin mapped |
| F4 | 0.650 | 0.65615 | 0.65615 | PI ±0.010 | Near peak (sharp) |
| F5 | 7575 | 7910 | 8088 | x1→0.995 | At ceiling |
| F6 | −0.297 | −0.297 | −0.297 | x1↓ (last lever) | At local optimum |
| F7 | 2.274 | 2.355 | 2.35702 | x3↓ | Converging |
| F8 | 9.622 | 9.708 | 9.772 | Trust-region GP | ACTIVE WINNER — climbing |
