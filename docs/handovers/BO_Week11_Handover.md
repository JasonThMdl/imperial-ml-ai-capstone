# BBO Capstone — Week 11 Handover

*Last updated: 2026-06-18 (entering Week 12)*

## Overview

Week 11 produced 3 new bests (F5, F7, F8). F5 hit its true ceiling with all dims at 0.999 (+330 vs previous best). F7 continued climbing from the W10 breakthrough (+0.069). F8 maintained its run (+0.039, now at 9.908). The peer-guided F3 probe at (0.43, 0.53, 0.46) returned −0.013, confirming our basin at (0.466, 0.490, 0.426) is correct and the peer's claimed −0.00006 does not transfer to our function instance. F4 GP candidate (0.408, 0.430, 0.358, 0.403) gave 0.591 — another near-miss; the true best (0.656) remains at (0.396, 0.418, 0.359, 0.414) and the W12 trust-region is centred there, not on the GP's candidate.

---

## Current best results (entering Week 12, after W11 results)

| Fn | Dim | n | Best y | Best x |
|----|-----|---|--------|--------|
| F1 | 2 | 21 | 0.0000 | flat |
| F2 | 2 | 21 | 0.6368 | (0.717, 0.969) |
| F3 | 3 | 26 | −0.00689 | (0.466, 0.490, 0.426) |
| F4 | 4 | 41 | 0.65615 | (0.396, 0.418, 0.359, 0.414) |
| F5 | 4 | 31 | 8585.27 | (0.999, 0.999, 0.999, 0.999) |
| F6 | 5 | 31 | −0.2967 | (0.325, 0.349, 0.446, 0.770, 0.010) |
| F7 | 6 | 41 | 2.76254 | (0.007260, 0.163677, 0.529682, 0.306376, 0.333504, 0.701637) |
| F8 | 8 | 51 | 9.90803 | (0.063148, 0.183640, 0.138555, 0.114332, 0.581725, 0.744996, 0.192960, 0.764036) |

---

## Week 11 results — scorecard

| Fn | W10 best | W11 result | New best? | Verdict |
|----|----------|------------|-----------|---------|
| F1 | 0.000 | 2.31e-29 | no | Flat confirmed; done |
| F2 | 0.6368 | 0.5879 | no | Re-submit was slightly off; (0.717, 0.969) remains best |
| F3 | −0.00689 | −0.01330 | no | Peer cluster (0.43, 0.53, 0.46) does not transfer — our basin is correct |
| F4 | 0.65615 | 0.5907 | no | GP candidate missed again; true best is still the W9 point |
| F5 | 8254.91 | 8585.27 | **yes** | All-0.999 ceiling confirmed; +330. Try 0.9995 next |
| F6 | −0.2967 | −0.3539 | no | Re-submit landed slightly off — our stored best coords are correct |
| F7 | 2.6933 | 2.7625 | **yes** | Trust-region continues delivering; +0.069 |
| F8 | 9.8691 | 9.9080 | **yes** | Compound run: 9.622→9.708→9.772→9.823→9.869→9.908 |

---

## Week 12 queries

| Fn | Query | Method | Rationale |
|----|-------|--------|-----------|
| F1 | 0.250441 − 0.993642 | Maximin | Space-fill; function is flat |
| F2 | 0.717000 − 0.969100 | Re-submit best | Converged; best point confirmed |
| F3 | 0.480643 − 0.495171 − 0.434706 | GP PI tight box (κ=0.5) | Peer probe failed; exploit known basin — minor nudge from (0.466, 0.490, 0.426) |
| F4 | 0.405787 − 0.427693 − 0.356063 − 0.416737 | Trust-region PI (κ=0.3) | Centred on TRUE best (0.396,0.418,0.359,0.414), not GP candidate |
| F5 | 0.999500 − 0.999500 − 0.999500 − 0.999500 | Boundary push | 0.999 improved on 0.995; test if 0.9995 gives further gain |
| F6 | 0.325074 − 0.348782 − 0.445668 − 0.770325 − 0.010000 | Re-submit best | All levers exhausted; −0.297 is the local optimum |
| F7 | 0.005740 − 0.191781 − 0.518238 − 0.272556 − 0.322607 − 0.669498 | Trust-region GP UCB (κ=1.5) | Tight box around W11 best; GP mu=2.744 — still rising |
| F8 | 0.052458 − 0.146474 − 0.146030 − 0.131916 − 0.635225 − 0.706067 − 0.176472 − 0.779459 | Trust-region GP UCB (κ=1.5) | Tight box around W11 best; GP mu=9.940 — predicted gain of +0.032 |

---

## Key notes entering Week 12

**F3 peer probe verdict:** −0.013 confirms the peer's −0.00006 does not apply to our function instance. Our basin at (0.466, 0.490, 0.426) is the correct focus. W12 uses a tiny PI nudge (κ=0.5, box ±0.015) to search for the true peak within the basin rather than re-submitting the exact same point.

**F4 root cause:** The GP keeps proposing candidates near (0.408, 0.430) that miss the true best (0.396, 0.418). The issue is the GP's surrogate is slightly misaligned near the sharp peak. W12 fixes this by centring the trust-region on the actual best-observed x, not the GP's argmax.

**F5 still open?** 0.999 improved on 0.995 (+330). If 0.9995 also improves it, the function is still rising at the boundary. If it plateaus or drops, 0.999 is the true ceiling.

**F8 GP prediction:** mu=9.940 at the W12 candidate (sd=0.007, very tight). This is the most calibrated prediction in the entire campaign — trust it.

---

## Progress tracker

| Fn | W8 | W9 | W10 | W11 | Status |
|----|----|----|----|-----|--------|
| F1 | 0.000 | 0.000 | 0.000 | 0.000 | Done |
| F2 | 0.6368 | 0.6368 | 0.6368 | 0.6368 | Done |
| F3 | −0.00689 | −0.00689 | −0.00689 | −0.00689 | Basin confirmed — micro-exploit |
| F4 | 0.65615 | 0.65615 | 0.65615 | 0.65615 | Sharp peak — re-centre trust-region |
| F5 | 8088 | 8180 | 8255 | **8585** | Ceiling at 0.999; test 0.9995 |
| F6 | −0.297 | −0.297 | −0.297 | −0.297 | Done |
| F7 | 2.357 | 2.357 | 2.693 | **2.763** | ACTIVE — trust-region climbing |
| F8 | 9.772 | 9.823 | 9.869 | **9.908** | ACTIVE WINNER — climbing steadily |
