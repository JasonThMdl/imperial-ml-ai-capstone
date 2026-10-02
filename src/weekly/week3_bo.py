"""Week 3 Bayesian Optimisation queries for the 8 black-box functions.

Implements per-function strategy as described in BO_Week2_Handover.docx, after
loading the W2 .npy files (each containing n+1 rows after the Week 2 result).
"""
import os
import sys
import warnings
import numpy as np
from scipy.stats import norm
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, RBF, WhiteKernel, ConstantKernel as C

sys.stdout.reconfigure(encoding="utf-8")
warnings.filterwarnings("ignore")

BASE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "data", "W2_data")
RNG = np.random.default_rng(42)


def load(i):
    X = np.load(os.path.join(BASE, f"function_{i}", f"f{i}_initial_inputs.npy"))
    y = np.load(os.path.join(BASE, f"function_{i}", f"f{i}_initial_outputs.npy"))
    return X, y


def fit_gp(X, y, kernel, alpha=1e-6, n_restarts=10):
    gp = GaussianProcessRegressor(
        kernel=kernel,
        alpha=alpha,
        normalize_y=True,
        n_restarts_optimizer=n_restarts,
        random_state=0,
    )
    gp.fit(X, y)
    return gp


def ucb(gp, X, kappa):
    mu, sigma = gp.predict(X, return_std=True)
    return mu + kappa * sigma, mu, sigma


def ei(gp, X, y_best, xi=0.0):
    mu, sigma = gp.predict(X, return_std=True)
    with np.errstate(divide="ignore", invalid="ignore"):
        imp = mu - y_best - xi
        z = imp / sigma
        out = imp * norm.cdf(z) + sigma * norm.pdf(z)
        out[sigma < 1e-12] = 0.0
    return out, mu, sigma


def thompson(gp, X, rng):
    mu, sigma = gp.predict(X, return_std=True)
    return rng.normal(mu, sigma), mu, sigma


def random_candidates(d, n, edge=0.005, rng=None):
    rng = rng or np.random.default_rng(0)
    return rng.uniform(edge, 1 - edge, size=(n, d))


def correlations(X, y):
    out = []
    for d in range(X.shape[1]):
        r = np.corrcoef(X[:, d], y)[0, 1]
        out.append(r)
    return out


def report(i, name, X, y, q, mu_q, sigma_q):
    top_idx = np.argsort(y)[-5:][::-1]
    print(f"\n=== F{i}  {name}  (n={len(y)}, d={X.shape[1]}) ===")
    print(f" Best y so far : {y.max():.4f} at {X[np.argmax(y)]}")
    print(f" Last y (W2)   : {y[-1]:.4f} at {X[-1]}")
    print(f" Top-5 y       : {y[top_idx]}")
    print(f" Correlations  : {[f'{c:+.2f}' for c in correlations(X, y)]}")
    print(f" Proposed query: {q}")
    print(f" GP at query   : mu={mu_q:.4f}  sigma={sigma_q:.4f}")
    return "-".join(f"{v:.6f}" for v in q)


submissions = {}


# ----------- F1: 2D · Flat, switch to Thompson sampling ----------------
i = 1
X, y = load(i)
# Outputs all zero — variance is nil, so any GP fit is degenerate. Use a
# Thompson-sampling-like approach on a Matern + WhiteKernel surrogate; if still
# flat we just sample a structured exploration point far from prior queries.
kernel = C(1.0, (1e-3, 1e3)) * Matern(length_scale=[0.2, 0.2], nu=2.5) + WhiteKernel(
    noise_level=1e-3, noise_level_bounds=(1e-6, 1e-1)
)
gp = fit_gp(X, y, kernel)
# Pick a candidate that is farthest (in L2) from current X — i.e. maximise the
# minimum distance to all observed points. This is essentially space-filling
# exploration when there's no signal.
cand = random_candidates(2, 20000, rng=np.random.default_rng(11))
mins = np.min(np.linalg.norm(cand[:, None, :] - X[None, :, :], axis=2), axis=1)
q = cand[np.argmax(mins)]
mu_q, sigma_q = gp.predict(q.reshape(1, -1), return_std=True)
submissions[i] = report(i, "Radiation (FLAT, exploration)", X, y, q, float(mu_q), float(sigma_q))


# ----------- F2: 2D · Noisy ML log-likelihood, explore lower-x1 -------
# Week 2 query (0.811, 0.925) gave 0.057 — much worse than best 0.6131 at
# (0.721, 0.970). Pushing x1 past 0.72 collapsed performance, so do NOT
# extrapolate further to higher x1. Per handover plan, "explore lower-x1
# region" and keep x2 high. Restrict candidates accordingly.
i = 2
X, y = load(i)
kernel = C(1.0, (1e-3, 1e3)) * Matern(length_scale=[0.2, 0.2], nu=2.5) + WhiteKernel(
    noise_level=1e-3, noise_level_bounds=(1e-6, 1e-1)
)
gp = fit_gp(X, y, kernel)
rng2 = np.random.default_rng(22)
cand = np.column_stack([
    rng2.uniform(0.30, 0.72, 40000),    # lower-x1 exploration
    rng2.uniform(0.85, 0.995, 40000),   # keep x2 high
])
acq, mu, sigma = ucb(gp, cand, kappa=2.0)
q = cand[np.argmax(acq)]
mu_q, sigma_q = gp.predict(q.reshape(1, -1), return_std=True)
submissions[i] = report(i, "Noisy ML (UCB κ=2.0, x1∈[0.30,0.72])", X, y, q, float(mu_q), float(sigma_q))


# ----------- F3: 3D · Non-smooth, tighten around best, x3 in [0.10, 0.35] -
i = 3
X, y = load(i)
kernel = C(1.0, (1e-3, 1e3)) * Matern(length_scale=[0.2, 0.2, 0.2], nu=0.5)
gp = fit_gp(X, y, kernel)
# Restrict candidate sampling to a tight box around the known best
# (0.493, 0.612, 0.340), since the W2 query at (0.7, 0.8, 0.158) failed
# (-0.11) — extrapolating away from best hurts on this non-smooth surface.
rng3 = np.random.default_rng(33)
cand = np.column_stack([
    rng3.uniform(0.35, 0.65, 50000),
    rng3.uniform(0.45, 0.75, 50000),
    rng3.uniform(0.25, 0.45, 50000),
])
acq, mu, sigma = ucb(gp, cand, kappa=2.5)
# Sanity guard: keep mu within the plausible range of top-observed
top_y = np.sort(y)[-5:]
mask = mu > (top_y.min() - 0.05)
if mask.any():
    cand_m, acq_m = cand[mask], acq[mask]
    q = cand_m[np.argmax(acq_m)]
else:
    q = cand[np.argmax(acq)]
mu_q, sigma_q = gp.predict(q.reshape(1, -1), return_std=True)
submissions[i] = report(i, "Drug (UCB κ=2.5, x3∈[0.10,0.35])", X, y, q, float(mu_q), float(sigma_q))


# ----------- F4: 4D · Multimodal, raise κ slightly --------------------
i = 4
X, y = load(i)
kernel = C(1.0, (1e-3, 1e3)) * Matern(length_scale=[0.2] * 4, nu=2.5)
gp = fit_gp(X, y, kernel)
cand = random_candidates(4, 60000, rng=np.random.default_rng(44))
acq, mu, sigma = ucb(gp, cand, kappa=2.5)
q = cand[np.argmax(acq)]
mu_q, sigma_q = gp.predict(q.reshape(1, -1), return_std=True)
submissions[i] = report(i, "Warehouse (UCB κ=2.5)", X, y, q, float(mu_q), float(sigma_q))


# ----------- F5: 4D · Unimodal smooth — EI ---------------------------
# Update: handover plan said "push x1 lower" based on initial r=-0.26, but the
# evidence after W1/W2 contradicts that — x1=0.265 → 4256.6 then x1=0.304 →
# 4270.4. Moving x1 HIGHER actually improved y. Widen x1 search range so EI
# can decide; x2/x3/x4 still pinned at boundary.
i = 5
X, y = load(i)
kernel = C(1.0, (1e-3, 1e3)) * RBF(length_scale=[0.2] * 4)
gp = fit_gp(X, y, kernel)
rng5 = np.random.default_rng(55)
cand = np.column_stack([
    rng5.uniform(0.05, 0.55, 80000),
    rng5.uniform(0.985, 0.995, 80000),
    rng5.uniform(0.985, 0.995, 80000),
    rng5.uniform(0.985, 0.995, 80000),
])
acq, mu, sigma = ei(gp, cand, y.max())
q = cand[np.argmax(acq)]
mu_q, sigma_q = gp.predict(q.reshape(1, -1), return_std=True)
submissions[i] = report(i, "Chemical (EI, x1 lower)", X, y, q, float(mu_q), float(sigma_q))


# ----------- F6: 5D · Reduce κ, exploit new best ----------------------
i = 6
X, y = load(i)
kernel = C(1.0, (1e-3, 1e3)) * Matern(length_scale=[0.2] * 5, nu=1.5)
gp = fit_gp(X, y, kernel)
best_idx = np.argmax(y)
best_x = X[best_idx]
rng6 = np.random.default_rng(66)
# Tighten around best: small box around best_x, x5 squeezed toward 0
cand = np.column_stack([
    rng6.uniform(max(0.005, best_x[0] - 0.20), min(0.995, best_x[0] + 0.20), 60000),
    rng6.uniform(max(0.005, best_x[1] - 0.20), min(0.995, best_x[1] + 0.20), 60000),
    rng6.uniform(max(0.005, best_x[2] - 0.20), min(0.995, best_x[2] + 0.20), 60000),
    rng6.uniform(0.70, 0.85, 60000),     # x4 closer to 0.8
    rng6.uniform(0.005, 0.08, 60000),    # x5 closer to 0
])
acq, mu, sigma = ucb(gp, cand, kappa=1.8)  # reduced from 2.5
q = cand[np.argmax(acq)]
mu_q, sigma_q = gp.predict(q.reshape(1, -1), return_std=True)
submissions[i] = report(i, "Cake (UCB κ=1.8, exploit)", X, y, q, float(mu_q), float(sigma_q))


# ----------- F7: 6D · raise κ to 1.8, explore x3/x4 -------------------
i = 7
X, y = load(i)
kernel = C(1.0, (1e-3, 1e3)) * Matern(length_scale=[0.2] * 6, nu=2.5)
gp = fit_gp(X, y, kernel)
rng7 = np.random.default_rng(77)
# Keep x1 near 0, x6 high, x5 moderate-low; let x3, x4 explore widely
cand = np.column_stack([
    rng7.uniform(0.005, 0.10, 60000),
    rng7.uniform(0.06, 0.20, 60000),    # near best x2≈0.10
    rng7.uniform(0.20, 0.80, 60000),    # explore x3 widely
    rng7.uniform(0.05, 0.60, 60000),    # explore x4 widely
    rng7.uniform(0.20, 0.45, 60000),
    rng7.uniform(0.80, 0.99, 60000),
])
acq, mu, sigma = ucb(gp, cand, kappa=1.8)
q = cand[np.argmax(acq)]
mu_q, sigma_q = gp.predict(q.reshape(1, -1), return_std=True)
submissions[i] = report(i, "ML hyperparam (UCB κ=1.8, explore x3/x4)", X, y, q, float(mu_q), float(sigma_q))


# ----------- F8: 8D · Manual override, x1<0.1, x3<0.1, x6/x8 high ------
i = 8
X, y = load(i)
kernel = C(1.0, (1e-3, 1e3)) * Matern(length_scale=[0.05, 0.2, 0.05, 0.2, 0.2, 0.2, 0.2, 0.2], nu=2.5)
gp = fit_gp(X, y, kernel)
rng8 = np.random.default_rng(88)
cand = np.column_stack([
    rng8.uniform(0.005, 0.08, 80000),     # x1 mandatory low
    rng8.uniform(0.005, 0.30, 80000),     # x2 lower (r≈-0.28)
    rng8.uniform(0.005, 0.08, 80000),     # x3 mandatory low
    rng8.uniform(0.005, 0.20, 80000),     # x4 lower
    rng8.uniform(0.25, 0.42, 80000),      # x5 moderate
    rng8.uniform(0.80, 0.95, 80000),      # x6 mandatory high
    rng8.uniform(0.005, 0.30, 80000),     # x7 lower (r≈-0.33)
    rng8.uniform(0.80, 0.95, 80000),      # x8 mandatory high
])
acq, mu, sigma = ucb(gp, cand, kappa=2.0)  # slight reduction from 2.5
q = cand[np.argmax(acq)]
mu_q, sigma_q = gp.predict(q.reshape(1, -1), return_std=True)
submissions[i] = report(i, "F8 (UCB κ=2.0, manual constraints)", X, y, q, float(mu_q), float(sigma_q))


print("\n\n========== WEEK 3 SUBMISSIONS ==========")
for i in range(1, 9):
    print(f"F{i}: {submissions[i]}")
