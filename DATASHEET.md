# Datasheet — BBO Capstone (Functions 1–8)

This datasheet covers all eight black-box functions in one document. Questions that differ by function are answered in per-function tables. Questions that apply to the whole campaign are answered once.

## Function overview

**1–3, 5. Function, scenario, dimensionality, output**

| Fn | Real-world scenario (course framing) | Dim | Output represents | Notes |
|----|--------------------------------------|-----|-------------------|-------|
| F1 | Radiation / contamination source detection | 2D | Detected signal strength | Returned ≈ 0 (≤ 1e-15) almost everywhere |
| F2 | Noisy ML model log-likelihood | 2D | Model log-likelihood score | Evaluations are noisy |
| F3 | Drug discovery (compound combination) | 3D | Negated adverse-reaction score, maximise toward 0 | Negative by design |
| F4 | Warehouse placement | 4D | Logistics performance score | Only a tiny region is positive |
| F5 | Chemical-process yield | 4D | Yield | Large positive values (10³–10⁴) |
| F6 | Cake recipe (ingredient mix) | 5D | Negated taste/quality loss, maximise toward 0 | Negative by design |
| F7 | ML hyperparameter tuning | 6D | Model performance score | |
| F8 | High-dimensional optimisation | 8D | Generic performance score | |

**4. Initial data points.** F1: 10, F2: 10, F3: 15, F4: 30, F5: 20, F6: 20, F7: 30, F8: 40.

## Nature of the data

**1. Structure.** For each function *k* there are two files per snapshot: `fk_initial_inputs.npy` with shape `(n, d)` (floats in `[0, 1]`) and `fk_initial_outputs.npy` with shape `(n,)` (real-valued). Initial shapes run from `(10, 2)` for F1 to `(40, 8)` for F8. Final shapes (after W11) run from `(21, 2)` to `(51, 8)`.

**2. Evolution.** Each week adds exactly **one** row per function, the query I submitted plus its returned output. Snapshots are cumulative (`data/initial_data` → `data/W1_data` … `data/W11_data`), so 88 new points were added in total. The sampling pattern moved from broad (weeks 1–3: high-κ UCB, space-filling) to strongly local (weeks 7–11: trust regions and tight boxes around each function's best). One exploration week (W10) deliberately probed new regions on the stalled functions.

**3. Noise.**
- **F2 is clearly noisy.** Four near-identical queries at ≈ (0.717, 0.969) returned 0.614, 0.637, 0.575 and 0.588 (std ≈ 0.027). Its best value is therefore partly a lucky draw, so I stopped trying to "improve" it and re-confirmed instead.
- **F4, F7 and F8 behave close to deterministically** at the scales queried. GP-fitted noise terms were tiny, and back-tested predictions matched outcomes within ±2σ.
- I could not test repeatability on every function, since repeat queries cost budget. The other functions were assumed to have low noise unless the data said otherwise.

**4. Landscape character**

| Fn | Character | Evidence |
|----|-----------|----------|
| F1 | Flat (or a peak too narrow to find) | 21 evaluations, all ≈ 0; GP σ ≈ 0 everywhere |
| F2 | Smooth ridge plus noise | Peak cluster near (0.71, 0.93–0.97), confirmed independently by peers |
| F3 | Non-smooth, with cliffs | LOO residual up to 10–18σ; small steps flip the output sharply |
| F4 | Sharply multimodal, one narrow good basin | All 30 initial points negative (best −4.03); deviations of ±0.05 drop to −1 to −2.6 |
| F5 | Smooth, monotone, maximum at the boundary | Output rises steadily with x1 when x2–x4 ≈ 1; maximum with all inputs = 0.999 |
| F6 | Rugged, interacting dimensions | Every one-variable change from the best made it worse |
| F7 | Smooth locally, razor-sharp in x3 | x3 = 0.27 or 0.65 collapses output; a new region opened in W10 |
| F8 | Smooth locally, steady gradient | Local GP calibrated (LOO max\|z\| ≈ 2.2); new best every week from W7 |

## Your optimisation strategy

**1–2. Methods and why**

| Fn | Final method | Why for this function |
|----|--------------|-----------------------|
| F1 | Maximin space-filling | No signal to exploit; filling the largest gap is the only rational query |
| F2 | GP EI in a tight box → re-confirm | Low-D, noisy; WhiteKernel models the noise; gains below the noise floor are meaningless |
| F3 | GP (Matérn ½) → manual local steps | Global GP failed the calibration check because of cliffs |
| F4 | GP **PI** in a shrinking box | PI is the most conservative acquisition, right for one narrow peak |
| F5 | 1D line search, then boundary push | Monotone signal; a surrogate added nothing |
| F6 | GP/NN UCB → DE → manual one-variable probes | Surrogates underfit (25 points in 5D) |
| F7 | GP EI razor box → **trust-region GP UCB** | Sharp in x3; local model accurate |
| F8 | NN ensemble → **trust-region GP UCB** | 8D with ≤ 51 points: a global model fails, a local one works (TuRBO) |

**3. Exploration vs exploitation.** Decided per function every week rather than globally:
- UCB κ was lowered after an improvement and raised after a miss (range 1.5–3.5).
- PI or EI was used where exploiting a known peak mattered most.
- Box or trust-region size was a second dial, with small boxes for sharp peaks.
- W10 was a planned exploration week for the stalled functions. Its results (all worse) were themselves evidence that the existing basins were the right ones.

**4. Strategy changes.** Yes, substantially. Weeks 1–3 used GP + UCB/EI everywhere. Weeks 4–6 tried an NN ensemble for high-D (dropped: it could not reproduce its own best training point), switched F4 to PI and F5 to line search, and tried DE for F6. Weeks 7–11 added the LOO calibration check, trust-region GPs (F8, later F7) and manual steps where the GP was flagged. Each change is logged with its reason in `docs/handovers/`.

## Data handling and preprocessing

1. **Input scaling.** Inputs were already in `[0, 1]^d`, so no rescaling was needed. Candidates were clipped 0.5% inward from the edges (0.005–0.995), except F5's final boundary push to 0.999.
2. **Surrogates.** Gaussian Processes (scikit-learn) throughout. A bootstrap ensemble of 15 MLPs (2×64 ReLU) was used in weeks 4–6. Polynomial and correlation checks were used as sanity checks.
3. **Surrogate preprocessing.** `normalize_y=True` (outputs differ in scale from 10⁻¹⁶ to 10⁴). Kernels: Matérn ν = ½, 3/2 or 5/2 by smoothness, with ARD length-scales. `WhiteKernel` noise was added for F2, F4, F6, F7 and F8. Hyperparameters were fitted by marginal likelihood with 10 restarts. Input warping (Beta-CDF) was tried on F3 and rejected.
4. **Outliers.** No points were deleted. Every observation is real feedback. Catastrophic values (e.g. F4 ≈ −4) were kept because they mark where *not* to go, and trust regions kept them from distorting local fits. One F1 query in W7 was mis-keyed (0.900 instead of 0.993). It was kept as a valid observation and the intended corner was queried later.

## Weekly iteration and learning

1. **How new data changed my understanding.** The early correlation heuristics were often wrong in detail. F5's initial "lower x1" signal reversed after two rounds. F6's "x5 as low as possible" overshot the optimum at x5 ≈ 0.01–0.05. F8's "x1, x3 at the floor" was beaten by values slightly above it. Each week's result was used to falsify or confirm one specific hypothesis.
2. **Local optima.** I detected them as repeated failure of every one-variable change around a best point (F6: x4 up, x2 down, x3 up and x1 down all worse), together with W10's exploration probes all scoring lower (F3, F4, F6).
3. **Most informative queries.**
   - Boundary points (F5 at 0.999).
   - The F4 "second basin" probe, whose −0.995 settled that question.
   - F2 repeats, which measured the noise.
   - F8's first trust-region query, which broke the plateau exactly as predicted.
4. **If I restarted:**
   - Run the LOO calibration check and the back-test from week 1, and choose surrogates on evidence.
   - Use trust regions earlier for the high-dimensional functions.
   - Transform outputs (log for F1, rank or warping for F4).
   - Fix an exploration budget up front instead of deciding weekly.

## Performance and results

**1–2. Best output and input**

| Fn | Best y | Input vector |
|----|--------|--------------|
| F1 | 7.7e-16 (≈ 0) | (0.731, 0.733), from the initial sample |
| F2 | 0.6368 | (0.717, 0.969) |
| F3 | −0.006885 | (0.466, 0.490, 0.426) |
| F4 | 0.656148 | (0.396, 0.418, 0.359, 0.414) |
| F5 | 8585.27 | (0.999, 0.999, 0.999, 0.999) |
| F6 | −0.296689 | (0.325, 0.349, 0.446, 0.770, 0.010) |
| F7 | 2.762543 | (0.00726, 0.16368, 0.52968, 0.30638, 0.33350, 0.70164) |
| F8 | 9.908026 | (0.06315, 0.18364, 0.13856, 0.11433, 0.58173, 0.74500, 0.19296, 0.76404) |

**3. Confidence of near-global optimum**
- **High:** F5 (maximum is at the domain boundary; the function is monotone). F2 (independent peer agreement; remaining gap is below the noise level).
- **Moderate:** F3 and F4 (basins mapped densely; exploration elsewhere scored worse, but the surfaces are sharp, so a better nearby point may be missed). F8 (still climbing at W11; the GP predicted a further +0.03, so this is *not* yet the maximum).
- **Low:** F1 (flat everywhere we looked; a narrow peak could exist). F6 (one basin only; high-dimensional, underexplored). F7 (a whole new region appeared as late as W10).

**4. Alignment with expectations.** Mostly aligned with the course descriptions: F2 was noisy, F3 non-smooth, F4 multimodal with sharp drops, and F5 smooth and unimodal. F1 ("radiation detection, may have two optima") was the surprise: no detectable signal at all.

## Ethical, practical and general considerations

1. **Real-world relevance.** The same loop sits behind hyperparameter tuning, A/B testing, materials and drug screening, process engineering and (in my own work) choosing which forecasting-model experiment to run next when each retrain is expensive.
2. **Synthetic limitations.** The functions are stationary, have bounded inputs and come without real units or constraints. Real problems add safety constraints, drifting conditions, batch evaluations and costs that vary by query. The "scenario" labels are framing only, so domain priors could not really be used.
3. **Scaling.** Trust-region GPs scale reasonably to tens of dimensions and hundreds of points. Exact GPs become slow beyond about 10⁴ points, where sparse GPs or ensembles are needed. The manual, human-in-the-loop parts (one-variable probes, weekly rule updates) do **not** scale and would need to be automated and audited.
4. **Pitfalls for future users.**
   - Do not trust surrogate uncertainty without a calibration check.
   - Do not read correlations as causal directions in interacting dimensions.
   - Do not chase improvements smaller than the noise.
   - Watch for input-entry errors (one occurred in W7).
   - Remember that "best found" is not "global optimum", especially in 5–8D with ≤ 51 points.
