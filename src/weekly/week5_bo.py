"""Week 5 Bayesian Optimisation queries.

Methodology changes from W4:
  F1  — Maximin space-filling (W4 gave -0.006; function has variation, keep searching)
  F2  — GP EI, tighter box (x1=0.714 in W4 too low; best at x1≈0.721)
  F3  — GP UCB, exploit new best (-0.018 at (0.463,0.452,0.375))
  F4  — GP + Probability of Improvement (PI), ±0.03 box — replaces EI for hairline region
  F5  — 1D line search, NO surrogate (clear x1 monotone signal; push x1 above 0.779)
  F6  — Differential Evolution on NN surrogate — replaces GP/NN UCB (3 weeks stuck)
  F7  — GP EI, return to x3≈0.525 (x3=0.65 in W4 confirmed too high)
  F8  — NN UCB, x1/x3 near but NOT at floor (0.005 in W4 hurt; best at 0.027/0.080)
"""

import os, sys, warnings
import numpy as np
from scipy.stats import norm
from scipy.optimize import minimize, differential_evolution
from sklearn.neural_network import MLPRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, RBF, WhiteKernel, ConstantKernel as C

sys.stdout.reconfigure(encoding="utf-8")
warnings.filterwarnings("ignore")

BASE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "data", "W4_data")
RNG = np.random.default_rng(42)


# ── helpers ────────────────────────────────────────────────────────────────────

def load(i):
    X = np.load(os.path.join(BASE, f"function_{i}", f"f{i}_initial_inputs.npy"))
    y = np.load(os.path.join(BASE, f"function_{i}", f"f{i}_initial_outputs.npy"))
    return X, y


def fit_gp(X, y, kernel, alpha=1e-6, n_restarts=10):
    gp = GaussianProcessRegressor(
        kernel=kernel, alpha=alpha, normalize_y=True,
        n_restarts_optimizer=n_restarts, random_state=0,
    )
    gp.fit(X, y)
    return gp


def ei_gp(gp, X, y_best, xi=0.0):
    mu, sigma = gp.predict(X, return_std=True)
    with np.errstate(divide="ignore", invalid="ignore"):
        imp = mu - y_best - xi
        z = imp / sigma
        out = imp * norm.cdf(z) + sigma * norm.pdf(z)
        out[sigma < 1e-12] = 0.0
    return out, mu, sigma


def pi_gp(gp, X, y_best, xi=0.005):
    """Probability of Improvement — more conservative than EI, stays near known peak."""
    mu, sigma = gp.predict(X, return_std=True)
    with np.errstate(divide="ignore", invalid="ignore"):
        z = (mu - y_best - xi) / sigma
        out = norm.cdf(z)
        out[sigma < 1e-12] = 0.0
    return out, mu, sigma


def ucb_gp(gp, X, kappa):
    mu, sigma = gp.predict(X, return_std=True)
    return mu + kappa * sigma, mu, sigma


def build_ensemble(X_train, y_train, n_models=15, seed=0):
    rng = np.random.default_rng(seed)
    n = len(y_train)
    models = []
    for k in range(n_models):
        idx = rng.choice(n, size=n, replace=True)
        m = MLPRegressor(
            hidden_layer_sizes=(64, 64), activation="relu",
            max_iter=3000, random_state=k, alpha=0.01,
            learning_rate_init=1e-3,
        )
        m.fit(X_train[idx], y_train[idx])
        models.append(m)
    return models


def ensemble_ucb_val(models, x, kappa=2.0):
    preds = np.array([m.predict(x.reshape(1, -1))[0] for m in models])
    return preds.mean() + kappa * preds.std()


def correlations(X, y):
    return [round(float(np.corrcoef(X[:, d], y)[0, 1]), 3) for d in range(X.shape[1])]


def report(i, name, X, y, q, mu_q=None, sigma_q=None):
    print(f"\n=== F{i}  {name}  (n={len(y)}, d={X.shape[1]}) ===")
    print(f"  Best y so far : {y.max():.6f} at {[f'{v:.4f}' for v in X[np.argmax(y)]]}")
    print(f"  Last y (W4)   : {y[-1]:.6f}")
    print(f"  Correlations  : {correlations(X, y)}")
    print(f"  Proposed query: {[f'{v:.6f}' for v in q]}")
    if mu_q is not None:
        print(f"  Surrogate at q: mu={mu_q:.4f}  sigma={sigma_q:.4f}")
    return "-".join(f"{v:.6f}" for v in q)


submissions = {}


# ── F1: 2D · Maximin space-filling ────────────────────────────────────────────
# W4 returned -0.006 — first non-zero result. Function is not uniformly flat.
# Continue covering unexplored space before drawing conclusions.
i = 1
X, y = load(i)
cand = RNG.uniform(0.005, 0.995, size=(50000, 2))
mins = np.min(np.linalg.norm(cand[:, None, :] - X[None, :, :], axis=2), axis=1)
q1 = cand[np.argmax(mins)]
submissions[i] = report(i, "Radiation (maximin, continue)", X, y, q1)


# ── F2: 2D · Noisy ML — GP EI, tightened around known best ───────────────────
# W4: x1=0.714 gave 0.571 < best 0.613 at x1=0.721. Best confirmed at x1≈0.721.
# Tighten box to x1∈[0.715,0.730], x2∈[0.965,0.975].
i = 2
X, y = load(i)
kernel = C(1.0, (1e-3, 1e3)) * Matern(length_scale=[0.15, 0.05], nu=2.5) + WhiteKernel(
    noise_level=1e-3, noise_level_bounds=(1e-6, 0.1)
)
gp = fit_gp(X, y, kernel)
rng2 = np.random.default_rng(22)
cand = np.column_stack([
    rng2.uniform(0.715, 0.730, 60000),
    rng2.uniform(0.965, 0.975, 60000),
])
acq, mu, sigma = ei_gp(gp, cand, y.max())
q2 = cand[np.argmax(acq)]
mu_q2, sig_q2 = gp.predict(q2.reshape(1, -1), return_std=True)
submissions[i] = report(i, "Noisy ML (GP EI, x1∈[0.715,0.730])", X, y, q2,
                        float(mu_q2), float(sig_q2))


# ── F3: 3D · Drug — GP UCB, exploit new best ─────────────────────────────────
# New best -0.018 at (0.463, 0.452, 0.375). Tighter box around it; x3 cap at 0.44.
i = 3
X, y = load(i)
best_x3 = X[np.argmax(y)]
kernel = C(1.0, (1e-3, 1e3)) * Matern(length_scale=[0.15, 0.15, 0.10], nu=0.5)
gp = fit_gp(X, y, kernel)
rng3 = np.random.default_rng(33)
cand = np.column_stack([
    rng3.uniform(max(0.005, best_x3[0] - 0.06), min(0.995, best_x3[0] + 0.06), 60000),
    rng3.uniform(max(0.005, best_x3[1] - 0.07), min(0.995, best_x3[1] + 0.07), 60000),
    rng3.uniform(0.33, 0.44, 60000),
])
acq, mu, sigma = ucb_gp(gp, cand, kappa=2.0)
q3 = cand[np.argmax(acq)]
mu_q3, sig_q3 = gp.predict(q3.reshape(1, -1), return_std=True)
submissions[i] = report(i, "Drug (GP UCB κ=2.0, exploit new best)", X, y, q3,
                        float(mu_q3), float(sig_q3))


# ── F4: 4D · Warehouse — GP + PI, ±0.03 tight box ───────────────────────────
# NEW METHODOLOGY: Probability of Improvement replaces EI.
# PI is more conservative — it penalises straying from the known peak, which is
# what we need for a hairline feasible region. EI was too willing to explore
# the flat/negative surroundings. ±0.03 vs W4's ±0.07.
i = 4
X, y = load(i)
best_x4 = X[np.argmax(y)]  # (0.408, 0.411, 0.347, 0.433)
kernel4 = C(1.0, (1e-3, 1e3)) * Matern(length_scale=[0.1] * 4, nu=1.5) + WhiteKernel(
    noise_level=0.1, noise_level_bounds=(1e-4, 1.0)
)
gp4 = fit_gp(X, y, kernel4, alpha=1e-4, n_restarts=10)
rng4 = np.random.default_rng(44)
bounds4_pi = [(max(0.005, best_x4[d] - 0.03), min(0.995, best_x4[d] + 0.03))
              for d in range(4)]
cand4 = np.column_stack([rng4.uniform(b[0], b[1], 80000) for b in bounds4_pi])
acq4, mu4, sigma4 = pi_gp(gp4, cand4, y.max(), xi=0.005)
q4 = cand4[np.argmax(acq4)]
mu_q4, sig_q4 = gp4.predict(q4.reshape(1, -1), return_std=True)
submissions[i] = report(i, "Warehouse (GP PI ±0.03 — new methodology)", X, y, q4,
                        float(mu_q4), float(sig_q4))


# ── F5: 4D · Chemical — 1D line search, NO surrogate ────────────────────────
# NEW METHODOLOGY: No GP/NN. Pure data-driven line search on the x1 axis.
# Every ceiling-pinned submission shows a clean monotone trend in x1:
#   x1=0.265→4257, 0.304→4270, 0.544→4486, 0.779→5538
# Strategy: fit a 1D polynomial to ceiling points and find the next x1 above
# the current maximum. Pin x2/x3/x4 at the values that produced the best y.
i = 5
X, y = load(i)

ceiling_mask = (X[:, 1] >= 0.985) & (X[:, 2] >= 0.985) & (X[:, 3] >= 0.985)
x1_c = X[ceiling_mask, 0]
y_c  = y[ceiling_mask]

# Fit degree-2 polynomial; extrapolate above the highest x1 tried so far
coeffs = np.polyfit(x1_c, y_c, deg=2)
x1_min_next = x1_c.max() + 0.01
x1_grid = np.linspace(x1_min_next, 0.995, 10000)
y_poly   = np.polyval(coeffs, x1_grid)
x1_next  = float(x1_grid[np.argmax(y_poly)])
# Safety: if poly is still monotone-increasing, argmax lands at 0.995 — cap at 0.92
# to preserve room for one more query in W6.
x1_next = min(x1_next, 0.920)

# Pin x2/x3/x4 at the ceiling values from the best point ever seen
best_c_idx = ceiling_mask.nonzero()[0][np.argmax(y_c)]
q5 = np.array([x1_next,
               float(X[best_c_idx, 1]),
               float(X[best_c_idx, 2]),
               float(X[best_c_idx, 3])])
print(f"\n[F5 line search] ceiling points: x1={np.round(x1_c,3)}, y={np.round(y_c,1)}")
print(f"[F5 line search] poly coeffs: {np.round(coeffs,1)} → next x1={x1_next:.4f}")
submissions[i] = report(i, "Chemical (1D line search — no surrogate)", X, y, q5)


# ── F6: 5D · Cake — Differential Evolution on NN surrogate ──────────────────
# NEW METHODOLOGY: replaces GP/NN UCB + L-BFGS-B.
# Three weeks stuck at -0.357. The root problem:
#   - Pinning x5 at floor (0.005) hurt twice (W3: -0.683, W4: -0.698).
#   - Best has x5=0.056; the negative correlation r=-0.679 was misleading us
#     to the hard floor when the true optimum is near x5≈0.05.
#   - L-BFGS-B gets trapped in local optima; DE is a global population-based
#     optimizer that naturally escapes them.
# DE maximises NN UCB over the full feasible region with x5 freed to [0.01,0.12].
i = 6
X, y = load(i)
models6 = build_ensemble(X, y, n_models=15, seed=66)

bounds6_de = [
    (0.15, 0.60),   # x1 — near-neutral correlation, broad search
    (0.10, 0.45),   # x2 — lower is better (r=-0.41)
    (0.30, 0.65),   # x3 — slight positive (r=+0.17), widen from W4
    (0.68, 0.92),   # x4 — higher is better (r=+0.59)
    (0.01, 0.12),   # x5 — sweet spot near 0.05, NOT at hard floor
]

def neg_nn_ucb6(x):
    return -ensemble_ucb_val(models6, np.array(x), kappa=2.0)

result6 = differential_evolution(
    neg_nn_ucb6,
    bounds6_de,
    maxiter=600,
    popsize=20,
    seed=66,
    tol=1e-9,
    mutation=(0.5, 1.0),
    recombination=0.9,
    polish=True,  # L-BFGS-B refinement step after DE converges
)
q6 = np.clip(result6.x, [b[0] for b in bounds6_de], [b[1] for b in bounds6_de])
preds6 = np.array([m.predict(q6.reshape(1, -1))[0] for m in models6])
print(f"\n[F6 DE] converged: {result6.success}  fun={result6.fun:.4f}  nfev={result6.nfev}")
submissions[i] = report(i, "Cake (Diff. Evolution on NN UCB — new methodology)", X, y, q6,
                        preds6.mean(), preds6.std())


# ── F7: 6D · ML hyperparam — GP EI, tight box, x3 corrected ─────────────────
# W4 x3=0.650 gave 1.472 < best 2.274. Confirmed: x3≈0.525 is the peak.
# Tight box around known best; do not let x3 stray outside [0.48, 0.57].
i = 7
X, y = load(i)
kernel7 = C(1.0, (1e-3, 1e3)) * Matern(length_scale=[0.05] * 6, nu=1.5) + WhiteKernel(
    noise_level=1e-3, noise_level_bounds=(1e-6, 0.1)
)
gp7 = fit_gp(X, y, kernel7, alpha=1e-5, n_restarts=10)
rng7 = np.random.default_rng(77)
bounds7 = [
    (0.005, 0.020),  # x1 near 0 (r=-0.53)
    (0.08, 0.14),    # x2
    (0.48, 0.57),    # x3 tightly around 0.525 — confirmed critical
    (0.15, 0.22),    # x4
    (0.32, 0.42),    # x5
    (0.74, 0.82),    # x6 high
]
cand7 = np.column_stack([rng7.uniform(b[0], b[1], 60000) for b in bounds7])
acq7, mu7, sigma7 = ei_gp(gp7, cand7, y.max())
q7 = cand7[np.argmax(acq7)]
mu_q7, sig_q7 = gp7.predict(q7.reshape(1, -1), return_std=True)
submissions[i] = report(i, "ML hyperparam (GP EI, x3∈[0.48,0.57])", X, y, q7,
                        float(mu_q7), float(sig_q7))


# ── F8: 8D · NN UCB, x1/x3 near-floor (not at floor) ────────────────────────
# W4: x1=x3=0.005 gave 9.507 < best 9.622 at x1=0.027, x3=0.080.
# Lesson: the floor is NOT optimal for x1/x3 — sweet spot is slightly above.
# Also try lower x2 [0.05,0.15] and lower x5 [0.20,0.30] as per handover.
i = 8
X, y = load(i)
bounds8 = [
    (0.010, 0.050),  # x1 — near floor, not at floor (best was 0.027)
    (0.05,  0.18),   # x2 — lower than W4's 0.35 (best was 0.291)
    (0.040, 0.110),  # x3 — near floor, not at floor (best was 0.080)
    (0.05,  0.25),   # x4
    (0.20,  0.32),   # x5 — lower than W4's 0.40 (best was 0.290)
    (0.930, 0.995),  # x6 high
    (0.05,  0.14),   # x7 low (r=-0.46)
    (0.80,  0.96),   # x8 high
]
models8 = build_ensemble(X, y, n_models=15, seed=88)
rng8 = np.random.default_rng(88)
cand8 = np.column_stack([rng8.uniform(b[0], b[1], 80000) for b in bounds8])
preds8_all = np.array([m.predict(cand8) for m in models8])
acq8 = preds8_all.mean(axis=0) + 2.0 * preds8_all.std(axis=0)
q8 = cand8[np.argmax(acq8)]
preds8_q = np.array([m.predict(q8.reshape(1, -1))[0] for m in models8])
submissions[i] = report(i, "F8 (NN UCB, x1/x3 near-floor, lower x2/x5)", X, y, q8,
                        preds8_q.mean(), preds8_q.std())


# ── Final output ───────────────────────────────────────────────────────────────
print("\n\n========== WEEK 5 SUBMISSIONS ==========")
for i in range(1, 9):
    print(f"F{i}: {submissions[i]}")
