"""Week 10 Bayesian Optimisation queries. Reads from W9_data.

DIRECTIVE: do not give up on any function — 5 weeks left, keep EXPLORING.
"Converged" only means a local optimum on what we have sampled. Several
functions could hide better regions (F4 is multimodal; F6/F7 are high-D and we
have only worked one basin). This week shifts the stalled functions from
exploitation to EXPLORATION (high-kappa UCB / new regions), pushes F5 to the
true boundary, and keeps exploiting F8 (the one engine still climbing).

  F1 — maximin space-filling (pure exploration; largest gap)
  F2 — GP UCB kappa=3 over the FULL space (look beyond the x1≈0.7 ridge)
  F3 — explore a NEW region: higher x1 (corr +0.26), lower x3 (corr -0.47)
  F4 — MULTIMODAL: hunt a second basin (high-kappa UCB, away from known peak)
  F5 — push x1 to the true boundary 0.999 (we self-capped at 0.995)
  F6 — explore weak dims (x1,x2,x3) holding the strong signals (x5 low, x4 high)
  F7 — explore weak dims (x2,x4,x5,x6) holding x1 low; try x6 higher
  F8 — keep exploiting the winner: tight trust-region GP UCB
"""

import os, sys, warnings
import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, WhiteKernel, ConstantKernel as C

sys.stdout.reconfigure(encoding="utf-8")
warnings.filterwarnings("ignore")

BASE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "data", "W9_data")
RNG = np.random.default_rng(42)


def load(i):
    X = np.load(os.path.join(BASE, f"function_{i}", f"f{i}_initial_inputs.npy"))
    y = np.load(os.path.join(BASE, f"function_{i}", f"f{i}_initial_outputs.npy"))
    return X, y


def fit_gp(X, y, kernel, alpha=1e-6, n_restarts=10):
    gp = GaussianProcessRegressor(kernel=kernel, alpha=alpha, normalize_y=True,
                                  n_restarts_optimizer=n_restarts, random_state=0)
    gp.fit(X, y); return gp


def ucb_gp(gp, X, kappa):
    mu, sigma = gp.predict(X, return_std=True); return mu + kappa * sigma, mu, sigma


def explore_ucb(i, X, y, kernel, bounds, kappa, seed, alpha=1e-6, n=120000):
    gp = fit_gp(X, y, kernel, alpha=alpha)
    rng = np.random.default_rng(seed)
    cand = np.column_stack([rng.uniform(b[0], b[1], n) for b in bounds])
    acq, mu, sigma = ucb_gp(gp, cand, kappa)
    j = np.argmax(acq)
    q = cand[j]
    muq, sdq = gp.predict(q.reshape(1, -1), return_std=True)
    return q, float(muq), float(sdq)


def report(i, name, X, y, q, mu_q=None, sigma_q=None):
    print(f"\n=== F{i}  {name}  (n={len(y)}, d={X.shape[1]}) ===")
    print(f"  Best y so far : {y.max():.6f} at {[f'{v:.4f}' for v in X[np.argmax(y)]]}")
    print(f"  Last y (W9)   : {y[-1]:.6f}")
    print(f"  Proposed query: {[f'{v:.6f}' for v in q]}")
    if mu_q is not None:
        print(f"  Surrogate at q: mu={mu_q:.4f}  sigma={sigma_q:.4f}  (explore: high sigma is intentional)")
    return "-".join(f"{v:.6f}" for v in q)


submissions = {}

# ── F1: maximin (pure exploration) ──────────────────────────────────────────
i = 1; X, y = load(i)
cand = RNG.uniform(0.005, 0.995, size=(150000, 2))
mins = np.min(np.linalg.norm(cand[:, None, :] - X[None, :, :], axis=2), axis=1)
q1 = cand[np.argmax(mins)]
print(f"[F1] maximin min-dist={mins.max():.3f}")
submissions[i] = report(i, "Radiation (maximin explore)", X, y, q1)

# ── F2: explore beyond the ridge — GP UCB kappa=3 over full space ────────────
i = 2; X, y = load(i)
k2 = C(1.0, (1e-3, 1e3)) * Matern(length_scale=[0.15, 0.10], nu=2.5) + WhiteKernel(0.02, (1e-4, 0.3))
q2, mu2, sd2 = explore_ucb(i, X, y, k2, [(0.005, 0.995), (0.005, 0.995)], kappa=3.0, seed=102)
submissions[i] = report(i, "Noisy ML (GP UCB κ=3 EXPLORE full space)", X, y, q2, mu2, sd2)

# ── F3: explore a NEW region — higher x1, lower x3 ──────────────────────────
i = 3; X, y = load(i)
k3 = C(1.0, (1e-3, 1e3)) * Matern(length_scale=[0.15, 0.15, 0.10], nu=0.5)
q3, mu3, sd3 = explore_ucb(i, X, y, k3,
                           [(0.45, 0.72), (0.40, 0.60), (0.28, 0.45)], kappa=3.0, seed=103)
submissions[i] = report(i, "Drug (GP UCB κ=3 EXPLORE higher x1 / lower x3)", X, y, q3, mu3, sd3)

# ── F4: MULTIMODAL — hunt a second basin (high-kappa UCB, away from peak) ────
i = 4; X, y = load(i)
k4 = C(1.0, (1e-3, 1e3)) * Matern(length_scale=[0.12] * 4, nu=1.5) + WhiteKernel(0.1, (1e-4, 1.0))
# explore a broad moderate region (avoid extreme corners where F4 is catastrophic)
q4, mu4, sd4 = explore_ucb(i, X, y, k4, [(0.15, 0.75)] * 4, kappa=3.0, seed=104, alpha=1e-4)
submissions[i] = report(i, "Warehouse (GP UCB κ=3 HUNT 2nd basin)", X, y, q4, mu4, sd4)

# ── F5: push x1 to the true boundary 0.999 (we self-capped at 0.995) ────────
i = 5; X, y = load(i)
mask = (X[:, 1] >= 0.985) & (X[:, 2] >= 0.985) & (X[:, 3] >= 0.985)
bc = mask.nonzero()[0][np.argmax(y[mask])]
q5 = np.array([0.999, float(X[bc, 1]), float(X[bc, 2]), float(X[bc, 3])])
print(f"\n[F5] ceiling x1 trajectory y={np.round(y[mask],0)} -> push x1 to boundary 0.999")
submissions[i] = report(i, "Chemical (line search, x1->0.999 boundary)", X, y, q5)

# ── F6: explore weak dims, hold strong signals (x5 low, x4 high) ────────────
i = 6; X, y = load(i)
k6 = C(1.0, (1e-3, 1e3)) * Matern(length_scale=[0.2] * 5, nu=2.5) + WhiteKernel(0.05, (1e-4, 0.5))
bounds6 = [(0.15, 0.50), (0.20, 0.50), (0.35, 0.58), (0.70, 0.88), (0.005, 0.03)]
q6, mu6, sd6 = explore_ucb(i, X, y, k6, bounds6, kappa=3.0, seed=106)
submissions[i] = report(i, "Cake (GP UCB κ=3 EXPLORE weak dims)", X, y, q6, mu6, sd6)

# ── F7: explore weak dims, hold x1 low; try x6 higher ───────────────────────
i = 7; X, y = load(i)
k7 = C(1.0, (1e-3, 1e3)) * Matern(length_scale=[0.15] * 6, nu=1.5) + WhiteKernel(1e-3, (1e-6, 0.1))
bounds7 = [(0.005, 0.03), (0.05, 0.20), (0.45, 0.55), (0.12, 0.27), (0.30, 0.45), (0.74, 0.90)]
q7, mu7, sd7 = explore_ucb(i, X, y, k7, bounds7, kappa=3.0, seed=107, alpha=1e-5)
submissions[i] = report(i, "ML hyperparam (GP UCB κ=3 EXPLORE, x6 higher)", X, y, q7, mu7, sd7)

# ── F8: keep exploiting the winner — tight trust-region GP UCB ──────────────
i = 8; X, y = load(i)
best8 = X[np.argmax(y)]
k8 = C(1.0, (1e-3, 1e3)) * Matern(length_scale=[0.08] * 8, nu=2.5) + WhiteKernel(1e-3, (1e-6, 0.1))
gp8 = fit_gp(X, y, k8, alpha=1e-5)
half = np.array([0.012, 0.06, 0.025, 0.05, 0.06, 0.04, 0.05, 0.06])
lo = np.clip(best8 - half, 0.005, 0.995); hi = np.clip(best8 + half, 0.005, 0.995)
rng8 = np.random.default_rng(108)
cand8 = np.column_stack([rng8.uniform(lo[d], hi[d], 120000) for d in range(8)])
acq8, _, _ = ucb_gp(gp8, cand8, kappa=1.5)
q8 = cand8[np.argmax(acq8)]
mu8, sd8 = gp8.predict(q8.reshape(1, -1), return_std=True)
submissions[i] = report(i, "F8 (trust-region GP UCB EXPLOIT winner)", X, y, q8, float(mu8), float(sd8))

print("\n\n========== WEEK 10 SUBMISSIONS ==========")
for i in range(1, 9):
    print(f"F{i}: {submissions[i]}")
