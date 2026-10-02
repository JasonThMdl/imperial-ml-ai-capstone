# Peer Insights — Week 11 (filtered for actionability)

## Directly actionable data points

### F3 — potential better basin
- Peer found y ≈ **−0.00006** (vs our best −0.00689, ~100× better) in region **x1≈0.43, x2≈0.53, x3≈0.46**
- Our nearest sample to that centroid is dist=0.051 at (0.466, 0.517, 0.426), y=−0.0077 — we have never precisely probed that coordinate
- GP UCB(κ=3) at peer centroid = 0.022 vs 0.018 at our known best → high uncertainty, warrants a probe
- **Action: change W11 F3 query to (0.430, 0.530, 0.460)**

### F2 — cluster confirmation
- Multiple peers independently confirm cluster at **(x1≈0.71, x2≈0.93)**
- Our 4th-best observation is exactly there: (0.703, 0.927) y=0.611
- Our best (0.717, 0.969) is in the same cluster but with x2 pushed higher — our ridge is consistent with theirs
- No action needed; our W11 re-submit is already correct

### F5 — ceiling pattern confirmed
- Peers universally confirm high output requires all dims near 1.0
- One peer found alternating [0.99, 0.01, 0.99, 0.01] works well — but that's a completely different basin; our all-high basin (8255) likely dominates
- Our all-0.999 push is correct

### F4 — second basin rejection confirmed
- A peer also tried a second-basin probe at (0.28, 0.40, ...) and got catastrophic results (confirms our −0.995 is typical)
- Multiple peers describe the only good region as the center cluster ~(0.40–0.55)
- No new information; our re-confirm strategy is correct

### F1 — log-space insight (minor)
- One peer used PowerTransform / log-scaling since F1 is radiation-like (inverse-square), found a peak at ~(0.472, ?)
- Our F1 is already at ≈0 (machine precision), so we're effectively at the optimum — no action needed
- Confirms F1 is done

### F6 — conflicting signals (discard)
- One peer claims x5 (Milk) must stay >0.05, but our best keeps x5=0.010 and achieves −0.297
- Ingredient labeling likely differs between submissions; peer data for F6 is not transferable
- No action

### F7 — conflicting signals (discard)
- One peer claims "x3 > 0.45 collapses performance" but our W10 breakthrough had x3=0.528 → y=2.693 (our best ever)
- Our own data directly contradicts this constraint; ignore it
- Trust our W10 result; continue trust-region exploitation

### F8 — strategy confirmation
- Multiple peers use trust-region / local refinement for their highest-dim functions
- GP R²=1.0 warning noted by one peer — relevant if n<5 good points, but F8 has 50 obs and steady improvement
- No change to strategy

---

## Strategy/method insights (from peers)

| Insight | Source | Relevant to us? |
|---------|--------|-----------------|
| ExtraTrees with upweighting for best points fixes under-prediction near optima | Peer 1 | Low — GP is well-calibrated (LOO z<3) |
| Sampling bias from "candidates near best point" inflates surrogate confidence | Peer 1 | Worth noting in reflection |
| Silhouette score as gate for clustering (threshold ≥0.22) | Peer 3 | Good framing for reflection |
| ATB override: if GP R²=1.0 and <5 good labels, submit exact known best | Peer 7 | Confirms our F1/F2/F3 re-submit logic |
| Removing Optuna HPO saved compute with <0.001 CV score difference | Peer 3 | Confirms we don't need HPO |
| EI vs UCB: UCB consistently explored empty space on some functions | Peer 6 | Confirms our UCB→PI switch for F4 |

---

## Updated W11 query list (one change from original)

| Fn | Query | Change from original? |
|----|-------|-----------------------|
| F1 | 0.663084 − 0.434111 | No |
| F2 | 0.717000 − 0.969100 | No |
| F3 | **0.430000 − 0.530000 − 0.460000** | **YES — probe peer cluster** |
| F4 | 0.407767 − 0.429800 − 0.357761 − 0.403423 | No |
| F5 | 0.999000 − 0.999000 − 0.999000 − 0.999000 | No |
| F6 | 0.325074 − 0.348782 − 0.445668 − 0.770325 − 0.010000 | No |
| F7 | 0.007260 − 0.163677 − 0.529682 − 0.306376 − 0.333504 − 0.701637 | No |
| F8 | 0.063148 − 0.183640 − 0.138555 − 0.114332 − 0.581725 − 0.744996 − 0.192960 − 0.764036 | No |
