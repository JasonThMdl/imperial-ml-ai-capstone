# Week 4 BBO Reflection — Module 12 (Neural Networks & Backpropagation)

## Context

After three rounds of Bayesian optimisation across eight unknown functions (2D to 8D), I switched from pure Gaussian Process surrogates to a **bootstrap MLP ensemble** for the higher-dimensional functions (F4–F8) in Week 4. The ensemble provides both a mean prediction and an uncertainty estimate (ensemble standard deviation), feeding a UCB or EI acquisition function. For refinement, I used `scipy` L-BFGS-B to maximise the acquisition function — which internally computes finite-difference gradients through the neural network's predictions, approximating what analytical backpropagation would give. The 2D/3D functions (F1–F3) retained GP surrogates where the GP inductive bias is clearly advantageous.

---

## Prompt responses

### 1. Support vectors and regions of rapid change

The most striking examples appeared in **F4 (multimodal warehouse, 4D)** and **F3 (non-smooth drug discovery, 3D)**.

For F4, the dataset contains only one point with y > 0 (best = 0.524 at roughly (0.41, 0.41, 0.35, 0.43)), while every other query has returned values ranging from −1 to −2.6. This isolated positive outcome is consistent with a very narrow "good" region surrounded by a steep drop — functionally similar to a support vector sitting right on the decision boundary between feasible and infeasible space. Week 3's UCB query landed at (0.45, 0.41, 0.22, 0.43) and returned −2.59, confirming that even a small deviation along x3 and x1 triggers a sharp transition. Recognising this means Week 4 must exploit in a very tight neighbourhood (±0.07 around the best point) rather than explore.

For F3, the output surface appears non-smooth — the correlations show x3 carrying a negative signal (r = −0.55) overall, yet both best points have x3 ≈ 0.34–0.40. This suggests a narrow ridge where x3 around 0.38–0.41 is locally optimal, but higher values (x3 > 0.46) deteriorate rapidly. Points just outside this ridge act as decision-boundary markers: the query at (0.70, 0.80, 0.16) in Week 2 collapsed to −0.11, flagging a rapid-change region.

---

### 2. Neural network gradients and output sensitivity

Rather than training a single NN and computing analytical backpropagation gradients, I built a **bootstrap ensemble of 15 MLPs** (two 64-neuron ReLU layers each, trained on bootstrapped subsets of the observations). The ensemble's output variance serves as the uncertainty signal.

For the gradient guidance step, I passed the best UCB candidate into `scipy.optimize.minimize` with method `L-BFGS-B`, which uses finite differences through the NN's `predict` function. In effect, this steps in the direction of the acquisition function's gradient with respect to the input — conceptually equivalent to gradient ascent via backpropagation, without needing to explicitly unroll the MLP's computation graph.

For **F5 (chemical process, 4D)**, the GP surrogate reveals a clear gradient signal: increasing x1 while keeping x2/x3/x4 near 1.0 drives output upward (Week 1: x1 = 0.265 → 4257; Week 2: x1 = 0.304 → 4270; Week 3: x1 = 0.544 → 4486). The surrogate's gradient in the x1 direction is strongly positive, suggesting continued improvement. Week 4 proposes x1 = 0.779. For **F8 (8D)**, the gradient signal is dominated by x1 and x3 (both r ≈ −0.69): the acquisition function gradient consistently points toward lower x1 and x3 simultaneously with higher x6.

---

### 3. Framing BBO as a classification task

This framing is most natural for **F4**, where the output is either positive (feasible/good, y > 0) or negative (infeasible/bad, y < 0). With 33 observations — 1 positive, 32 negative — a support vector classifier (SVM) with a radial basis function kernel could potentially delineate the narrow good region. The trade-off: a hard-margin SVM on such imbalanced data might over-fit the single positive example, while a soft-margin SVM would likely classify the entire input space as "bad" given the class imbalance.

Logistic regression would draw a linear boundary, which is likely too rigid for a multimodal 4D surface. A neural network classifier, even a small one, could in principle learn the curved boundary between the y > 0 pocket and the surrounding negative region — but with only one positive training example, any classifier will generalise poorly without strong regularisation.

The practical approach I took is equivalent to soft classification: the EI acquisition function implicitly treats "better than y_best" as the positive class, and EI naturally handles the imbalance by weighting the probability of improvement.

---

### 4. Model choice: interpretability vs. flexibility

For the **2D functions (F1, F2)** and **3D function (F3)**, Gaussian processes remained the right tool. With 12–18 observations and clear input correlations, the GP's kernel structure (Matérn ν = 2.5 for smooth functions, ν = 0.5 for non-smooth) imposes a meaningful inductive bias and gives well-calibrated uncertainty. Interpretability is high: you can read the length-scale hyperparameters directly as measures of effective smoothness.

For the **5–8D functions (F5–F8)**, the neural network ensemble becomes valuable because:
- The GP's kernel must parameterise over a higher-dimensional covariance structure, making hyperparameter optimisation fragile
- The MLP's hidden layers can represent interaction effects between inputs (e.g., the observation that x5 being low is only beneficial for F6 when x4 is simultaneously high)
- The bootstrap ensemble resamples the small dataset to diversify the uncertainty signal

The main cost is calibration: with 23–43 points and 15 models, ensemble variance can be poorly calibrated in unexplored regions. F4 illustrated this sharply — the NN ensemble produced sigma = 6.6 in the tight exploitation region, compared to the GP's sigma = 0.62 for the same candidates. GP uncertainty was more reliable here, so I reverted F4 to a GP surrogate.

---

### 5. Steepest gradients and variable importance

Looking at the correlation-based gradient proxies across all functions:

| Function | Most important input | Direction |
|----------|---------------------|-----------|
| F2 | x2 | High (x2 ≥ 0.94 critical; dropping to 0.87 collapsed y by 0.06) |
| F5 | x2, x3, x4 (r ≈ 0.60–0.65) | Near ceiling; x1 rising beneficial |
| F6 | x5 (r = −0.66), x4 (r = +0.57) | x5 near 0; x4 high |
| F7 | x1 (r = −0.53), x6 (r = +0.43) | x1 near 0; x6 high |
| F8 | x1, x3 (r = −0.69 each) | Both near 0; dominate output |

For **F8**, the gradient signal from x1 and x3 is symmetric and strong — the function appears to penalise both dimensions jointly when they exceed roughly 0.10. This guided the Week 4 constraint x1 < 0.06 and x3 < 0.06, tighter than the Week 3 constraint of < 0.08. The NN ensemble UCB with these constraints produced a query at (0.005, 0.35, 0.005, 0.25, 0.40, 0.99, 0.05, 0.96), pushing x1 and x3 to the minimum permitted, and x6 and x8 to their ceilings.

---

### 6. Neural network decision boundary for F4

Framing F4 as a binary classification task (y > 0 vs y ≤ 0), a small NN trained on the 33 observations would place its decision boundary around the single cluster of input values near (0.41, 0.41, 0.35, 0.43). Because the boundary is compact and surrounded on all sides by negatives, backpropagation would drive the boundary to be a small hypersphere around the known positive point.

In practice, the NN ensemble (trained in regression mode) showed this implicitly: the ensemble standard deviation was high everywhere except very close to the known best, confirming it "knew" only that localised region was reliable. The gradient of the UCB acquisition function pointed back toward (0.41, 0.41, 0.35, 0.43), consistent with a NN that has effectively learned the positive-class centre.

Backpropagation was useful here not for finding new optima, but for confirming the tightness of the feasible region — the gradient magnitude drops rapidly as you move away from the best point, which is exactly what a decision-boundary classifier would show.

---

### 7. Non-linear patterns: NN vs. linear regression vs. GP

**F5 (chemical process)** is a useful comparison case. Linear regression on all 23 observations gives x1 a negative coefficient (r = −0.18), suggesting higher x1 reduces y. Yet the empirical trajectory shows the opposite: x1 = 0.54 gave the best result. This contradiction arises because the training set includes many observations with high x1 but low x2/x3/x4, which pull the linear fit in the wrong direction. A neural network — even a shallow one — can represent the multiplicative interaction "x1 × (x2, x3, x4 near ceiling)" that the linear model cannot.

For **F6** and **F7**, the non-linearity is subtler: the output depends on a specific combination of multiple dimensions simultaneously, not on any single dimension in isolation. A linear model identifies the strongest individual correlates but misses the interaction. The MLP ensemble, by contrast, can in principle approximate the interaction surface — though with only 23–33 points, its advantage over a GP with a well-chosen kernel is modest.

Overall assessment: the added flexibility of a neural network surrogate is most valuable for functions with clear interaction effects (F5, F6, F7) and least valuable where the function is either flat (F1), very noisy (F2), or has an extremely narrow feasible region (F4). The GP remains preferable when sample sizes are small (n < ~25) because its calibration is better grounded in the kernel prior.

---

## Summary of Week 4 queries

| Fn | Query | Predicted improvement |
|----|-------|-----------------------|
| F1 | 0.419725 − 0.463133 | Unknown (function flat) |
| F2 | 0.714350 − 0.969777 | GP predicts 0.615 vs current best 0.613 |
| F3 | 0.462875 − 0.451708 − 0.375416 | Tight exploitation near new best |
| F4 | 0.402554 − 0.384562 − 0.381604 − 0.414273 | GP EI near only positive point |
| F5 | 0.778636 − 0.994733 − 0.994727 − 0.990020 | GP predicts 4858 vs 4486 |
| F6 | 0.220000 − 0.318954 − 0.380000 − 0.850000 − 0.005000 | Corrects W3 failure; x5 pinned low |
| F7 | 0.005000 − 0.070000 − 0.650000 − 0.130000 − 0.342616 − 0.880000 | x3 corrected to 0.65 |
| F8 | 0.005000 − 0.350000 − 0.005000 − 0.250000 − 0.400000 − 0.990000 − 0.050000 − 0.960000 | Tighter constraints; progressive improvement expected |
