"""Week 6 Bayesian Optimisation queries.

Reads from W5_data. Strategy per function, informed by W5 outcomes:
  F1  — MANUAL: submit pre-planned maximin point "W13" (user choice)
  F2  — GP EI, tightest box yet around x1≈0.721 (x1=0.715→0.598 still below best)
  F3  — GP UCB, exploit new best -0.0077; raising x2/x3 helped, allow slight push up
  F4  — GP PI, tighten to ±0.025 around new best 0.566 (PI worked in W5)
  F5  — 1D line search, push x1 higher (trend accelerating: 5538→6984 at x1=0.92)
  F6  — Differential Evolution again (broke 3-week deadlock: -0.357→-0.297)
  F7  — GP EI, very tight around best; x3∈[0.51,0.535] (x3=0.541→2.244 just missed)
  F8  — NN UCB, tight EXPLOIT box around W3 best (two explore attempts regressed)
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

BASE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "data", "W5_data")
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
    """Probability of Improvement — conservative, stays near known peak."""
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
    print(f"  Last y (W5)   : {y[-1]:.6f}")
    print(f"  Correlations  : {correlations(X, y)}")
    print(f"  Proposed query: {[f'{v:.6f}' for v in q]}")
    if mu_q is not None:
        print(f"  Surrogate at q: mu={mu_q:.4f}  sigma={sigma_q:.4f}")
    return "-".join(f"{v:.6f}" for v in q)


submissions = {}


# ── F1: 2D · MANUAL — pre-planned maximin point "W13" ────────────────────────
# User decision: submit the W13 point from the 10-week maximin plan now.
# Point lies in the sparsely-sampled left-central region.
i = 1
X, y = load(i)
q1 = np.array([0.224981, 0.568320])
submissions[i] = report(i, "Radiation (MANUAL maximin point W13)", X, y, q1)


# ── F2: 2D · Noisy ML — GP EI, tightest box around x1≈0.721 ──────────────────
# W5: x1=0.715→0.598. Pattern x1=0.714→0.571, 0.715→0.598, 0.721→0.613(best).
# Closer to 0.721 is better. Box x1∈[0.718,0.726], x2∈[0.965,0.975].
i = 2
X, y = load(i)
kernel = C(1.0, (1e-3, 1e3)) * Matern(length_scale=[0.15, 0.05], nu=2.5) + WhiteKernel(
    noise_level=1e-3, noise_level_bounds=(1e-6, 0.1)
)
gp = fit_gp(X, y, kernel)
rng2 = np.random.default_rng(122)
cand = np.column_stack([
    rng2.uniform(0.718, 0.726, 60000),
    rng2.uniform(0.965, 0.975, 60000),
])
acq, mu, sigma = ei_gp(gp, cand, y.max())
q2 = cand[np.argmax(acq)]
mu_q2, sig_q2 = gp.predict(q2.reshape(1, -1), return_std=True)
submissions[i] = report(i, "Noisy ML (GP EI, x1∈[0.718,0.726])", X, y, q2,
                        float(mu_q2), float(sig_q2))


# ── F3: 3D · Drug — GP UCB, exploit new best, allow slight x2/x3 push ────────
# New best -0.0077 at (0.466, 0.517, 0.426). Raising x2 (0.452→0.517) and
# x3 (0.375→0.426) both improved y → the local peak may be slightly higher
# still. Allow x2 up to 0.58, x3 up to 0.47 (corr -0.506, so cap there).
i = 3
X, y = load(i)
best3 = X[np.argmax(y)]
kernel = C(1.0, (1e-3, 1e3)) * Matern(length_scale=[0.15, 0.15, 0.10], nu=0.5)
gp = fit_gp(X, y, kernel)
rng3 = np.random.default_rng(133)
cand = np.column_stack([
    rng3.uniform(0.42, 0.51, 60000),   # around x1≈0.466
    rng3.uniform(0.48, 0.58, 60000),   # x2: allow push above 0.517
    rng3.uniform(0.40, 0.47, 60000),   # x3: allow push above 0.426 (cap 0.47)
])
acq, mu, sigma = ucb_gp(gp, cand, kappa=1.5)
q3 = cand[np.argmax(acq)]
mu_q3, sig_q3 = gp.predict(q3.reshape(1, -1), return_std=True)
submissions[i] = report(i, "Drug (GP UCB κ=1.5, exploit + slight push)", X, y, q3,
                        float(mu_q3), float(sig_q3))


# ── F4: 4D · Warehouse — GP PI, tighten to ±0.025 around new best ───────────
# PI worked: W5 hit new best 0.566 at (0.405, 0.401, 0.369, 0.423).
# Tighten further now that we have a confirmed positive neighbourhood.
i = 4
X, y = load(i)
best4 = X[np.argmax(y)]
kernel4 = C(1.0, (1e-3, 1e3)) * Matern(length_scale=[0.1] * 4, nu=1.5) + WhiteKernel(
    noise_level=0.1, noise_level_bounds=(1e-4, 1.0)
)
gp4 = fit_gp(X, y, kernel4, alpha=1e-4, n_restarts=10)
rng4 = np.random.default_rng(144)
bounds4 = [(max(0.005, best4[d] - 0.025), min(0.995, best4[d] + 0.025))
           for d in range(4)]
cand4 = np.column_stack([rng4.uniform(b[0], b[1], 80000) for b in bounds4])
acq4, mu4, sigma4 = pi_gp(gp4, cand4, y.max(), xi=0.005)
q4 = cand4[np.argmax(acq4)]
mu_q4, sig_q4 = gp4.predict(q4.reshape(1, -1), return_std=True)
submissions[i] = report(i, "Warehouse (GP PI ±0.025 around new best)", X, y, q4,
                        float(mu_q4), float(sig_q4))


# ── F5: 4D · Chemical — 1D line search, push x1 higher ──────────────────────
# Trajectory accelerating: x1=0.779→5538, x1=0.920→6984 (Δy/Δx1 now ~10000).
# Function still climbing hard. Push x1 toward 0.96; pin x2/x3/x4 at ceiling.
i = 5
X, y = load(i)
ceiling_mask = (X[:, 1] >= 0.985) & (X[:, 2] >= 0.985) & (X[:, 3] >= 0.985)
x1_c = X[ceiling_mask, 0]
y_c = y[ceiling_mask]
coeffs = np.polyfit(x1_c, y_c, deg=2)
x1_grid = np.linspace(x1_c.max() + 0.005, 0.995, 10000)
x1_next = float(x1_grid[np.argmax(np.polyval(coeffs, x1_grid))])
x1_next = min(x1_next, 0.960)  # leave headroom for W7+ if still rising
best_c_idx = ceiling_mask.nonzero()[0][np.argmax(y_c)]
q5 = np.array([x1_next, float(X[best_c_idx, 1]), float(X[best_c_idx, 2]), float(X[best_c_idx, 3])])
print(f"\n[F5 line search] ceiling x1={np.round(x1_c,3)}  y={np.round(y_c,0)}")
print(f"[F5 line search] next x1={x1_next:.4f}")
submissions[i] = report(i, "Chemical (1D line search, x1→0.96)", X, y, q5)


# ── F6: 5D · Cake — MANUAL one-variable gradient step ───────────────────────
# The NN ensemble is underfit here: it predicts -0.64 at the known best whose
# true value is -0.30, i.e. it cannot even reproduce a training point (25 pts
# in 5D, bootstrap drops the rare good point). So its mean is uninformative and
# UCB just chases variance to box corners. We abandon the surrogate for F6.
#
# Data-driven instead: the W5 breakthrough (-0.357 -> -0.297) came from raising
# x4 (0.746 -> 0.770) and lowering x5 (0.056 -> 0.010). x4 is the strongest
# positive signal (r=+0.59). It was tried at 0.84-0.85 before but those points
# had x3 wrong (0.27, 0.38) and failed -> "high x4 WITH correct x3" is untested.
# Clean experiment: take the known best, change ONLY x4 (0.770 -> 0.80) — a step
# beyond the proven value, short of the 0.84 that failed. Isolates the x4 effect.
i = 6
X, y = load(i)
best6 = X[np.argmax(y)].copy()   # (0.3251, 0.3488, 0.4457, 0.7703, 0.0100)
q6 = best6.copy()
q6[3] = 0.80                      # push x4 up, hold all else at best
submissions[i] = report(i, "Cake (MANUAL: best, x4 0.770->0.80)", X, y, q6)


# ── F7: 6D · ML hyperparam — GP EI, very tight around best ──────────────────
# W5: x3=0.541→2.244, just below best 2.274 at x3=0.525. x3 is razor-sharp.
# Narrow x3 to [0.51,0.535] and tighten every other dim around the best.
i = 7
X, y = load(i)
best7 = X[np.argmax(y)]  # (0.005, 0.101, 0.525, 0.182, 0.365, 0.771)
kernel7 = C(1.0, (1e-3, 1e3)) * Matern(length_scale=[0.05] * 6, nu=1.5) + WhiteKernel(
    noise_level=1e-3, noise_level_bounds=(1e-6, 0.1)
)
gp7 = fit_gp(X, y, kernel7, alpha=1e-5, n_restarts=10)
rng7 = np.random.default_rng(177)
bounds7 = [
    (0.005, 0.012),  # x1 near 0
    (0.085, 0.115),  # x2 around 0.101
    (0.510, 0.535),  # x3 — razor tight around 0.525
    (0.165, 0.200),  # x4 around 0.182
    (0.340, 0.390),  # x5 around 0.365
    (0.755, 0.795),  # x6 around 0.771
]
cand7 = np.column_stack([rng7.uniform(b[0], b[1], 60000) for b in bounds7])
acq7, mu7, sigma7 = ei_gp(gp7, cand7, y.max())
q7 = cand7[np.argmax(acq7)]
mu_q7, sig_q7 = gp7.predict(q7.reshape(1, -1), return_std=True)
submissions[i] = report(i, "ML hyperparam (GP EI, x3∈[0.510,0.535])", X, y, q7,
                        float(mu_q7), float(sig_q7))


# ── F8: 8D · NN UCB, tight EXPLOIT box around W3 best ───────────────────────
# Two explore attempts both regressed (W4=9.507, W5=9.581 < best 9.622).
# Stop exploring. Tight exploitation box around the W3 best to try to edge
# above 9.622. Best = (0.027,0.291,0.080,0.181,0.290,0.940,0.126,0.804).
i = 8
X, y = load(i)
best8 = X[np.argmax(y)]
deltas8 = [0.015, 0.05, 0.025, 0.04, 0.05, 0.025, 0.03, 0.04]
bounds8 = [(max(0.005, best8[d] - deltas8[d]), min(0.995, best8[d] + deltas8[d]))
           for d in range(8)]
models8 = build_ensemble(X, y, n_models=15, seed=188)
rng8 = np.random.default_rng(188)
cand8 = np.column_stack([rng8.uniform(b[0], b[1], 80000) for b in bounds8])
preds8_all = np.array([m.predict(cand8) for m in models8])
acq8 = preds8_all.mean(axis=0) + 1.5 * preds8_all.std(axis=0)  # lower kappa = exploit
q8 = cand8[np.argmax(acq8)]
preds8_q = np.array([m.predict(q8.reshape(1, -1))[0] for m in models8])
submissions[i] = report(i, "F8 (NN UCB κ=1.5, tight exploit around best)", X, y, q8,
                        preds8_q.mean(), preds8_q.std())


# ── Final output ───────────────────────────────────────────────────────────────
print("\n\n========== WEEK 6 SUBMISSIONS ==========")
for i in range(1, 9):
    print(f"F{i}: {submissions[i]}")
