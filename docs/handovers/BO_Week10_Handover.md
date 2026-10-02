# BBO Capstone — Week 10 Handover

*Last updated: 2026-06-16 (entering Week 11)*

## Overview

Week 10 was an **exploration week** for most functions — deliberate high-kappa UCB probes away from known peaks. The exploration of F3/F4/F6 found nothing better (confirming existing basins are likely optimal). The headline wins came from the functions where we stayed targeted: **F5 hit a new high at the true x1 boundary (0.999)**, **F7 had a major breakthrough (+0.336 via x6 higher)**, and **F8 continued its run (+0.046)**. 3 new bests on the week.

---

## Current best results (entering Week 11, after W10 results)

| Fn | Dim | n | Best y | Best x |
|----|-----|---|--------|--------|
| F1 | 2 | 20 | 0.0000 | flat |
| F2 | 2 | 20 | 0.6368 | (0.717, 0.969) — noise-driven, converged |
| F3 | 3 | 25 | −0.00689 | (0.466, 0.490, 0.426) |
| F4 | 4 | 40 | 0.65615 | (0.396, 0.418, 0.359, 0.414) |
| F5 | 4 | 30 | 8254.91 | (0.999, 0.994733, 0.994727, 0.990020) |
| F6 | 5 | 30 | −0.2967 | (0.325, 0.349, 0.446, 0.770, 0.010) |
| F7 | 6 | 40 | 2.69325 | (0.008709, 0.199615, 0.527814, 0.266869, 0.307563, 0.741346) |
| F8 | 8 | 50 | 9.86912 | (0.051782, 0.198991, 0.126358, 0.122438, 0.522981, 0.782972, 0.196365, 0.813655) |

---

## Week 10 results — scorecard

| Fn | W9 best | W10 result | New best? | Verdict |
|----|---------|------------|-----------|---------|
| F1 | 0.000 | ~0 (1e-152) | no | Flat confirmed; done |
| F2 | 0.6368 | −0.041 | no | Exploration found nothing; ridge near (0.7, 0.97) is the global peak |
| F3 | −0.00689 | −0.0661 | no | Higher x1 / lower x3 region is worse; basin at x≈(0.466,0.490,0.426) confirmed optimal |
| F4 | 0.65615 | −0.9951 | no | Second-basin hunt failed hard; the peak at ~0.656 is likely the only good region |
| F5 | 8180.10 | 8254.91 | **yes** | True boundary x1=0.999 > 0.995; +74.8. F5 now fully exhausted |
| F6 | −0.2967 | −0.3654 | no | Explored (x1≈0.46, x2≈0.43, x3≈0.58) — worse; confirms −0.297 is the local optimum |
| F7 | 2.35702 | 2.69325 | **yes** | Major jump: x6 higher (0.741) + x2 up (0.20) + x3 up (0.528) opened new region. +0.336 |
| F8 | 9.82284 | 9.86912 | **yes** | Trust-region GP UCB delivers again. Cumulative: 9.622 → 9.708 → 9.772 → 9.823 → 9.869 |

---

## Week 11 queries planned

Load from **W10_data**. Adjust a copy of `week10_bo.py`.

| Fn | Plan | Method | Rationale |
|----|------|--------|-----------|
| F1 | Maximin space-fill | Maximin | Flat; fill remaining gap as a courtesy query |
| F2 | Re-submit best point | Hold | Confirmed converged; noise-floor ~0.14 |
| F3 | Re-submit best point | Hold | Basin fully mapped; −0.00689 is likely optimal |
| F4 | Re-confirm best ±0.010 | GP PI tight box | 2nd-basin failed; the 0.656 peak is the global best. Re-confirm to rule out narrow miss |
| F5 | Push x2/x3/x4 ceiling | Line search → 0.999 | x1 exhausted; now push remaining dims to 0.999 |
| F6 | Re-submit best point | Hold | All levers probed; −0.297 is the local optimum |
| F7 | **Exploit the F7 breakthrough** | Trust-region GP UCB κ=1.5 | 2.693 is a major new region — mine it with a tight trust region centred on the W10 best |
| F8 | Continue trust-region GP UCB | Trust-region GP UCB κ=1.5 | Cumulative +0.247 over 4 weeks; still climbing — do not stop |

---

## Key insights from Week 10

**F7 breakthrough is the week's story.** Moving x6 from 0.77 to 0.74 while also shifting x2 (0.11→0.20), x3 (0.49→0.53), and x5 (0.36→0.31) produced a +0.336 jump. This is the largest single-week gain on F7 so far. The function almost certainly has a richer landscape in the region (x1≈0.01, x2≈0.20, x3≈0.53, x6≈0.74) than we thought. Week 11 must exploit this with a tight trust-region GP.

**F5 ceiling confirmed.** x1=0.999 > x1=0.995 as predicted. The gain was +74.8. The remaining x2/x3/x4 dims sit at ~0.990–0.995; if F5 rises toward all-0.999, there could be another ~50–100 units still available.

**F4 second-basin hypothesis rejected.** The −0.995 result at (0.28, 0.40, 0.36, 0.42) rules out another high-value basin there. The sharp peak at ≈0.656 is likely the global optimum for F4.

**F8 is the compound winner.** Five straight weeks of improvement; the trust-region GP is correctly calibrated and has never over-predicted. Prediction-vs-actual tracking is valuable — maintain it.

---

## Conditional Week 11 strategy (short form)

- **F1/F2/F3/F6** — converged. Submit best point or a space-filler; no query investment.
- **F4** — re-confirm the 0.656 peak with a tight ±0.010 PI box. Accept it if nothing better found.
- **F5** — push x2/x3/x4 toward 0.999. Check if function continues rising or plateaus.
- **F7** — **TOP PRIORITY.** New region opened. Tight trust-region GP UCB centred on (0.009, 0.200, 0.528, 0.267, 0.308, 0.741). Half-widths ≈ 0.02 on active dims, 0.05 on weaker.
- **F8** — continue trust-region GP UCB. Do not widen the box; it is working.

---

## Progress tracker

| Fn | W7 | W8 | W9 | W10 | Status |
|----|----|----|----|-----|--------|
| F1 | 0.000 | 0.000 | 0.000 | 0.000 | Flat — done |
| F2 | 0.6136 | 0.6368* | 0.6368* | 0.6368* | Converged (noise) — stop |
| F3 | −0.00689 | −0.00689 | −0.00689 | −0.00689 | Basin mapped — stop |
| F4 | 0.65615 | 0.65615 | 0.65615 | 0.65615 | Sharp peak — re-confirm only |
| F5 | 7910 | 8088 | 8180 | **8255** | True ceiling found — push x2/x3/x4 |
| F6 | −0.297 | −0.297 | −0.297 | −0.297 | Local optimum — stop |
| F7 | 2.355 | 2.357 | 2.357 | **2.693** | BREAKTHROUGH — exploit W11 |
| F8 | 9.708 | 9.772 | 9.823 | **9.869** | ACTIVE WINNER — keep going |
