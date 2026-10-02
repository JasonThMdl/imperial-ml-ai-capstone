# BBO Capstone — Week 5 Handover

*Reconstructed 2026-06-04 for numbering consistency (covers Week 5 submission → Week 6 plan).*

## Overview

Week 5 was the round with the biggest methodology shifts. Three functions changed approach based on accumulated evidence: F4 switched from Expected Improvement to **Probability of Improvement** (PI is more conservative for its hairline feasible region), F5 dropped the surrogate entirely for a **1D line search** (clean monotone x1 signal), and F6 moved from GP/NN UCB to **Differential Evolution** after three weeks stuck. Queries submitted via `week5_bo.py` (reads from `W4_data`). This document records the bests entering Week 6, the W5 queries submitted, and the conditional Week 6 strategy.

---

## Current best results (entering Week 6, after W4 results)

| Fn | Dim | n | Best y | Best x |
|----|-----|---|--------|--------|
| F1 | 2 | 14 | 0.0000 | flat — all queries ≈ 0 (one −0.006 in W4) |
| F2 | 2 | 14 | 0.6131 | (0.721, 0.970) |
| F3 | 3 | 19 | −0.0177 | (0.463, 0.452, 0.375) — new best in W4 |
| F4 | 4 | 34 | 0.5235 | (0.408, 0.411, 0.347, 0.433) |
| F5 | 4 | 24 | 5538.39 | (0.779, 0.995, 0.995, 0.990) — new best in W4 |
| F6 | 5 | 24 | −0.3572 | (0.361, 0.338, 0.456, 0.746, 0.056) |
| F7 | 6 | 34 | 2.2742 | (0.005, 0.101, 0.525, 0.182, 0.365, 0.771) |
| F8 | 8 | 44 | 9.6220 | (0.027, 0.291, 0.080, 0.181, 0.290, 0.940, 0.126, 0.804) |

---

## Week 4 results — scorecard

| Fn | W3 best | W4 result | New best? | Verdict |
|----|---------|-----------|-----------|---------|
| F1 | 0.000 | −0.006 | no | First non-zero (negative); function not perfectly flat |
| F2 | 0.613 | 0.571 | no | x1=0.714 too low vs best at 0.721 |
| F3 | −0.0287 | −0.0177 | **yes** | Meaningful improvement |
| F4 | 0.524 | 0.305 | no | Positive but below best; narrow region confirmed |
| F5 | 4486 | 5538 | **yes** | Beat GP prediction (4857); still rising |
| F6 | −0.357 | −0.698 | no | x5=0.005 floor-pinning hurt |
| F7 | 2.274 | 1.472 | no | x3=0.650 far too high |
| F8 | 9.622 | 9.507 | no | Small regression |

Score: 2 clear wins (F3, F5), rest misses → motivated the W5 methodology changes.

---

## Week 5 queries submitted

| Fn | Query | Method | Rationale |
|----|-------|--------|-----------|
| F1 | 0.008712-0.011769 | Maximin | Continue space-filling; W4 showed function has slight variation |
| F2 | 0.715000-0.972413 | GP EI | Tighten toward proven x1≈0.721, x2≥0.965 |
| F3 | 0.465650-0.517222-0.425813 | GP UCB | Exploit the new W4 best |
| F4 | 0.405124-0.401175-0.368953-0.422615 | **GP PI** ±0.03 | *New:* PI replaces EI — conservative for hairline region |
| F5 | 0.920000-0.994733-0.994727-0.990020 | **1D line search** | *New:* drop surrogate; x1 monotone trend, push to 0.92 |
| F6 | 0.325074-0.348782-0.445668-0.770325-0.010000 | **Differential Evolution** | *New:* global optimiser; free x5 from hard floor |
| F7 | 0.005296-0.081775-0.541021-0.169480-0.346338-0.773383 | GP EI | Return x3 to ≈0.525 (W4 x3=0.65 was wrong) |
| F8 | 0.010571-0.179322-0.059790-0.051867-0.317242-0.978413-0.100443-0.957703 | NN UCB | x1/x3 near floor but not at it (best was 0.027/0.080) |

---

## Key lessons from Week 4 → Week 5

- **F5** — the trajectory (4257 → 4270 → 4486 → 5538 as x1 rises) is a learned 1D signal; no surrogate needed. Line search is the right tool.
- **F6** — pinning x5 to the hard floor (0.005) hurt twice. The negative correlation (r=−0.66) was misleading us past the true optimum at x5≈0.05. GP/NN UCB + L-BFGS-B kept getting stuck → switch to Differential Evolution (global, population-based).
- **F4** — EI was too willing to explore the negative surroundings of the single positive point. PI penalises straying from the known peak → better for a hairline feasible region.
- **F7** — x3 is razor-sharp: dropping it to 0.27 (W3) and raising it to 0.65 (W4) both failed. x3 ≈ 0.525 is the peak.
- **F8** — the hard floor is *not* optimal for x1/x3; the best point sits slightly above it (0.027, 0.080). Stop pushing to 0.005.

---

## Week 6 strategy (conditional on W5 results)

Load from **W5_data**. Adjust a copy of `week5_bo.py`.

- **F1** — keep searching (10 queries left, 2D space is coverable). Pre-plan remaining maximin points. If any query returns > 0, switch to GP UCB immediately.
- **F2** — if W5 > 0.613, exploit the new point; else tighten to x1∈[0.718,0.726], x2∈[0.965,0.975].
- **F3** — if W5 improved, the upward x2/x3 push is working; continue. Else exploit tightly around the best.
- **F4** — if W5 > 0.524, tighten PI box to ±0.025 around the new best. Keep PI, not EI.
- **F5** — if still rising at x1=0.92, push x1 toward 0.96. If it drops, plateau reached → tight box around best x1.
- **F6** — if DE improved on −0.357, exploit that region but watch for the ensemble chasing variance to box corners (it is underfit at this n). Consider a manual one-variable step instead.
- **F7** — if W5 > 2.274, tighten around it; else narrow to x3∈[0.51,0.535] and fine-tune the rest.
- **F8** — if W5 > 9.622, exploit; else 9.622 (W3) is robust — vary one untested dimension at a time.

---

## Progress tracker

| Fn | W3 best | W4 best | W5 query | Trend |
|----|---------|---------|----------|-------|
| F1 | 0.000 | 0.000 | Maximin | Flat |
| F2 | 0.613 | 0.613 | Exploit | Stable |
| F3 | −0.029 | −0.018 | Exploit | Improving |
| F4 | 0.524 | 0.524 | PI (new) | Stalled — fragile |
| F5 | 4486 | 5538 | Line search (new) | Strong uptrend |
| F6 | −0.357 | −0.357 | DE (new) | Was stalled |
| F7 | 2.274 | 2.274 | x3 corrected | Stalled |
| F8 | 9.622 | 9.622 | Near-floor | Steady |
