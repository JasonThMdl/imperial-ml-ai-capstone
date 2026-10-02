"""Week 9 Bayesian Optimisation queries. Reads from W8_data.

W8 gave 4 new bests (F2-noise, F5, F7, F8). Strategy:
  F1 — maximin; bottom-right corner is the last big gap (top-right now filled)
  F2 — converged & noisy (0.6368 was a noise draw); re-confirm the best point
  F3 — basin bracketed (x2 opt ~0.49); probe last lever x3 DOWN (corr -0.47)
  F4 — sharp peak; tighten PI to ±0.010 to re-converge (W8 0.6448 < 0.6561)
  F5 — push x1 to 0.995 (ceiling) — still rising (+178 at 0.99), final value
  F6 — x4-up, x2-down, x3-up all failed; probe the LAST clean lever x1 DOWN
  F7 — x3 down still edging up (2.357); continue x3 0.503 -> 0.49
  F8 — exploit the breakthrough; tight trust-region GP UCB (9.62->9.71->9.77)
"""

import os, sys, warnings
import numpy as np
from scipy.stats import norm
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, WhiteKernel, ConstantKernel as C

sys.stdout.reconfigure(encoding="utf-8")
warnings.filterwarnings("ignore")

BASE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "data", "W8_data")
RNG = np.random.default_rng(42)


def load(i):
    X = np.load(os.path.join(BASE, f"function_{i}", f"f{i}_initial_inputs.npy"))
    y = np.load(os.path.join(BASE, f"function_{i}", f"f{i}_initial_outputs.npy"))
    return X, y


def fit_gp(X, y, kernel, alpha=1e-6, n_restarts=10):
    gp = GaussianProcessRegressor(kernel=kernel, alpha=alpha, normalize_y=True,
                                  n_restarts_optimizer=n_restarts, random_state=0)
    gp.fit(X, y); return gp


def pi_gp(gp, X, y_best, xi=0.005):
    mu, sigma = gp.predict(X, return_std=True)
    with np.errstate(divide="ignore", invalid="ignore"):
        z = (mu - y_best - xi) / sigma; out = norm.cdf(z); out[sigma < 1e-12] = 0.0
    return out, mu, sigma


def ucb_gp(gp, X, kappa):
    mu, sigma = gp.predict(X, return_std=True); return mu + kappa * sigma, mu, sigma


def loo_diagnostic(tag, X, y, kernel, alpha=1e-6):
    n = len(y); z = np.zeros(n)
    for k in range(n):
        m = np.ones(n, bool); m[k] = False
        g = GaussianProcessRegressor(kernel=kernel, alpha=alpha, normalize_y=True,
                                     n_restarts_optimizer=3, random_state=0)
        g.fit(X[m], y[m]); mu, sd = g.predict(X[k:k+1], return_std=True)
        z[k] = (y[k] - mu[0]) / sd[0]
    flag = "  ⚠ |z|>3" if np.abs(z).max() > 3 else "  ✓ calibrated"
    print(f"   [LOO {tag}] std(z)={z.std():.2f}  max|z|={np.abs(z).max():.1f}{flag}")


def report(i, name, X, y, q, mu_q=None, sigma_q=None):
    print(f"\n=== F{i}  {name}  (n={len(y)}, d={X.shape[1]}) ===")
    print(f"  Best y so far : {y.max():.6f} at {[f'{v:.4f}' for v in X[np.argmax(y)]]}")
    print(f"  Last y (W8)   : {y[-1]:.6f}")
    print(f"  Proposed query: {[f'{v:.6f}' for v in q]}")
    if mu_q is not None:
        print(f"  Surrogate at q: mu={mu_q:.4f}  sigma={sigma_q:.4f}")
    return "-".join(f"{v:.6f}" for v in q)


submissions = {}

# ── F1: maximin (bottom-right corner now the largest gap) ────────────────────
i = 1; X, y = load(i)
cand = RNG.uniform(0.005, 0.995, size=(120000, 2))
mins = np.min(np.linalg.norm(cand[:, None, :] - X[None, :, :], axis=2), axis=1)
q1 = cand[np.argmax(mins)]
print(f"[F1] maximin min-dist={mins.max():.3f} at ({q1[0]:.4f},{q1[1]:.4f})")
submissions[i] = report(i, "Radiation (maximin, largest current gap)", X, y, q1)

# ── F2: converged & noisy — re-confirm the all-time best point ───────────────
i = 2; X, y = load(i)
q2 = X[np.argmax(y)].copy()  # the recorded best location
submissions[i] = report(i, "Noisy ML (re-confirm best; CONVERGED)", X, y, q2)

# ── F3: probe last lever — x3 DOWN (corr -0.47), basin bracketed at x2~0.49 ──
i = 3; X, y = load(i)
best3 = X[np.argmax(y)].copy(); q3 = best3.copy(); q3[2] = 0.400
submissions[i] = report(i, "Drug (MANUAL: x3 0.426->0.400, last lever)", X, y, q3)

# ── F4: sharp peak — tighten PI to ±0.010 to re-converge ────────────────────
i = 4; X, y = load(i)
best4 = X[np.argmax(y)]
kernel4 = C(1.0, (1e-3, 1e3)) * Matern(length_scale=[0.1] * 4, nu=1.5) + WhiteKernel(
    noise_level=0.1, noise_level_bounds=(1e-4, 1.0))
loo_diagnostic("F4", X, y, kernel4, alpha=1e-4)
gp4 = fit_gp(X, y, kernel4, alpha=1e-4)
rng4 = np.random.default_rng(449)
bounds4 = [(max(0.005, best4[d] - 0.010), min(0.995, best4[d] + 0.010)) for d in range(4)]
cand4 = np.column_stack([rng4.uniform(b[0], b[1], 90000) for b in bounds4])
acq4, _, _ = pi_gp(gp4, cand4, y.max(), xi=0.003)
q4 = cand4[np.argmax(acq4)]
mu_q4, sig_q4 = gp4.predict(q4.reshape(1, -1), return_std=True)
submissions[i] = report(i, "Warehouse (GP PI ±0.010, re-converge)", X, y, q4, float(mu_q4), float(sig_q4))

# ── F5: line search, push x1 to the ceiling 0.995 ───────────────────────────
i = 5; X, y = load(i)
mask = (X[:, 1] >= 0.985) & (X[:, 2] >= 0.985) & (X[:, 3] >= 0.985)
bc = mask.nonzero()[0][np.argmax(y[mask])]
q5 = np.array([0.995, float(X[bc, 1]), float(X[bc, 2]), float(X[bc, 3])])
print(f"\n[F5] ceiling x1={np.round(X[mask,0],3)} y={np.round(y[mask],0)} -> push to 0.995 (ceiling)")
submissions[i] = report(i, "Chemical (line search, x1->0.995 ceiling)", X, y, q5)

# ── F6: probe LAST clean lever — x1 DOWN (corr -0.19); others falsified ──────
i = 6; X, y = load(i)
best6 = X[np.argmax(y)].copy(); q6 = best6.copy(); q6[0] = 0.280
submissions[i] = report(i, "Cake (MANUAL: x1 0.325->0.280, last lever)", X, y, q6)

# ── F7: continue x3 DOWN (proven direction) 0.503 -> 0.49 ───────────────────
i = 7; X, y = load(i)
best7 = X[np.argmax(y)].copy(); q7 = best7.copy(); q7[2] = 0.490
submissions[i] = report(i, "ML hyperparam (MANUAL: x3 0.503->0.490)", X, y, q7)

# ── F8: exploit breakthrough — tight trust-region GP UCB ────────────────────
i = 8; X, y = load(i)
best8 = X[np.argmax(y)]
kernel8 = C(1.0, (1e-3, 1e3)) * Matern(length_scale=[0.08] * 8, nu=2.5) + WhiteKernel(
    noise_level=1e-3, noise_level_bounds=(1e-6, 0.1))
loo_diagnostic("F8", X, y, kernel8, alpha=1e-5)
gp8 = fit_gp(X, y, kernel8, alpha=1e-5)
half = np.array([0.012, 0.06, 0.025, 0.05, 0.06, 0.04, 0.05, 0.06])
lo = np.clip(best8 - half, 0.005, 0.995); hi = np.clip(best8 + half, 0.005, 0.995)
rng8 = np.random.default_rng(890)
cand8 = np.column_stack([rng8.uniform(lo[d], hi[d], 120000) for d in range(8)])
acq8, _, _ = ucb_gp(gp8, cand8, kappa=1.5)
q8 = cand8[np.argmax(acq8)]
mu_q8, sig_q8 = gp8.predict(q8.reshape(1, -1), return_std=True)
submissions[i] = report(i, "F8 (trust-region GP UCB κ=1.5, exploit)", X, y, q8, float(mu_q8), float(sig_q8))

print("\n\n========== WEEK 9 SUBMISSIONS ==========")
for i in range(1, 9):
    print(f"F{i}: {submissions[i]}")
