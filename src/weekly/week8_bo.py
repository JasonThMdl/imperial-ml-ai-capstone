"""Week 8 Bayesian Optimisation queries.

Reads from W7_data. Strategies derived + adversarially verified by an 8-function
workflow (analyze -> verify), then synthesised here. W7 gave 5 new bests.

  F1 — maximin (recomputed on 17 obs); top-right is the largest gap now
  F2 — converged & noisy; re-sample cluster CENTROID (verifier: avoid the exact
       0.717/0.965 spot that misfired to 0.531 in W7) -> (0.717, 0.9691)
  F3 — continue the x2-DOWN gradient that produced the W7 best (override the
       workflow's weak x3-up idea, which fights x3's negative correlation)
  F4 — GP PI, tight box around new best 0.6561 (calibrated locally)
  F5 — line search; still rising (7910 at x1=0.98), push x1 -> 0.99
  F6 — manual one-var: x3 0.446 -> 0.453 (verifier tightened from 0.460); the
       only clean untested lever (x4-up and x2-down already falsified)
  F7 — manual one-var: continue x3 DOWN 0.515 -> 0.503 (x3=0.515 beat 0.525)
  F8 — trust-region GP UCB around the breakthrough best 9.7083 (LOO z=1.9 ✓)
"""

import os, sys, warnings
import numpy as np
from scipy.stats import norm
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, WhiteKernel, ConstantKernel as C

sys.stdout.reconfigure(encoding="utf-8")
warnings.filterwarnings("ignore")

BASE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "data", "W7_data")
RNG = np.random.default_rng(42)


def load(i):
    X = np.load(os.path.join(BASE, f"function_{i}", f"f{i}_initial_inputs.npy"))
    y = np.load(os.path.join(BASE, f"function_{i}", f"f{i}_initial_outputs.npy"))
    return X, y


def fit_gp(X, y, kernel, alpha=1e-6, n_restarts=10):
    gp = GaussianProcessRegressor(kernel=kernel, alpha=alpha, normalize_y=True,
                                  n_restarts_optimizer=n_restarts, random_state=0)
    gp.fit(X, y)
    return gp


def pi_gp(gp, X, y_best, xi=0.005):
    mu, sigma = gp.predict(X, return_std=True)
    with np.errstate(divide="ignore", invalid="ignore"):
        z = (mu - y_best - xi) / sigma
        out = norm.cdf(z); out[sigma < 1e-12] = 0.0
    return out, mu, sigma


def ucb_gp(gp, X, kappa):
    mu, sigma = gp.predict(X, return_std=True)
    return mu + kappa * sigma, mu, sigma


def loo_diagnostic(tag, X, y, kernel, alpha=1e-6):
    n = len(y); z = np.zeros(n)
    for k in range(n):
        m = np.ones(n, bool); m[k] = False
        gp = GaussianProcessRegressor(kernel=kernel, alpha=alpha, normalize_y=True,
                                      n_restarts_optimizer=3, random_state=0)
        gp.fit(X[m], y[m]); mu, sd = gp.predict(X[k:k+1], return_std=True)
        z[k] = (y[k] - mu[0]) / sd[0]
    flag = "  ⚠ |z|>3 not trustworthy" if np.abs(z).max() > 3 else "  ✓ calibrated"
    print(f"   [LOO {tag}] std(z)={z.std():.2f}  max|z|={np.abs(z).max():.1f}{flag}")


def correlations(X, y):
    return [round(float(np.corrcoef(X[:, d], y)[0, 1]), 3) for d in range(X.shape[1])]


def report(i, name, X, y, q, mu_q=None, sigma_q=None):
    print(f"\n=== F{i}  {name}  (n={len(y)}, d={X.shape[1]}) ===")
    print(f"  Best y so far : {y.max():.6f} at {[f'{v:.4f}' for v in X[np.argmax(y)]]}")
    print(f"  Last y (W7)   : {y[-1]:.6f}")
    print(f"  Proposed query: {[f'{v:.6f}' for v in q]}")
    if mu_q is not None:
        print(f"  Surrogate at q: mu={mu_q:.4f}  sigma={sigma_q:.4f}")
    return "-".join(f"{v:.6f}" for v in q)


submissions = {}


# ── F1: 2D · Maximin (recompute on 17 obs) ───────────────────────────────────
i = 1
X, y = load(i)
cand = RNG.uniform(0.005, 0.995, size=(120000, 2))
mins = np.min(np.linalg.norm(cand[:, None, :] - X[None, :, :], axis=2), axis=1)
q1 = cand[np.argmax(mins)]
# also report the still-unexplored mis-keyed corner as an alternative
print(f"[F1] maximin best min-dist={mins.max():.3f} at ({q1[0]:.4f},{q1[1]:.4f}); "
      f"mis-keyed corner (0.9933,0.0057) still unexplored")
submissions[i] = report(i, "Radiation (maximin, largest current gap)", X, y, q1)


# ── F2: 2D · Noisy — re-sample high-value cluster centroid ───────────────────
# Converged at ~0.6136; noise std~0.035. Verifier: do NOT re-hit (0.717,0.965)
# which misfired to 0.531 in W7. Centroid of the good replicates = (0.717,0.9691).
i = 2
X, y = load(i)
q2 = np.array([0.717000, 0.969100])
submissions[i] = report(i, "Noisy ML (re-sample cluster centroid; converged)", X, y, q2)


# ── F3: 3D · Drug — continue the x2-DOWN gradient (surrogate untrusted) ───────
# W7 best came from x2 0.517->0.49. Continue down to 0.47 (cliff caution).
# (Override of workflow's x3-up, which contradicts x3 corr=-0.47.)
i = 3
X, y = load(i)
best3 = X[np.argmax(y)].copy()  # (0.4657, 0.490, 0.4258)
q3 = best3.copy(); q3[1] = 0.470
submissions[i] = report(i, "Drug (MANUAL: continue x2 0.490->0.470)", X, y, q3)


# ── F4: 4D · Warehouse — GP PI, tight box around new best 0.6561 ─────────────
i = 4
X, y = load(i)
best4 = X[np.argmax(y)]
kernel4 = C(1.0, (1e-3, 1e3)) * Matern(length_scale=[0.1] * 4, nu=1.5) + WhiteKernel(
    noise_level=0.1, noise_level_bounds=(1e-4, 1.0))
loo_diagnostic("F4", X, y, kernel4, alpha=1e-4)
gp4 = fit_gp(X, y, kernel4, alpha=1e-4)
rng4 = np.random.default_rng(448)
bounds4 = [(max(0.005, best4[d] - 0.018), min(0.995, best4[d] + 0.018)) for d in range(4)]
cand4 = np.column_stack([rng4.uniform(b[0], b[1], 90000) for b in bounds4])
acq4, _, _ = pi_gp(gp4, cand4, y.max(), xi=0.005)
q4 = cand4[np.argmax(acq4)]
mu_q4, sig_q4 = gp4.predict(q4.reshape(1, -1), return_std=True)
submissions[i] = report(i, "Warehouse (GP PI ±0.018 around new best)", X, y, q4, float(mu_q4), float(sig_q4))


# ── F5: 4D · Chemical — line search, push x1 -> 0.99 ─────────────────────────
i = 5
X, y = load(i)
mask = (X[:, 1] >= 0.985) & (X[:, 2] >= 0.985) & (X[:, 3] >= 0.985)
bc = mask.nonzero()[0][np.argmax(y[mask])]
q5 = np.array([0.990, float(X[bc, 1]), float(X[bc, 2]), float(X[bc, 3])])
print(f"\n[F5] ceiling x1={np.round(X[mask,0],3)} y={np.round(y[mask],0)}  -> push to 0.99")
submissions[i] = report(i, "Chemical (line search, x1->0.99)", X, y, q5)


# ── F6: 5D · Cake — manual one-var: x3 0.446 -> 0.453 (verifier-tightened) ───
# x4-up and x2-down both falsified; x3 is the only clean untested lever.
i = 6
X, y = load(i)
best6 = X[np.argmax(y)].copy()  # (0.325, 0.349, 0.446, 0.770, 0.010)
q6 = best6.copy(); q6[2] = 0.453
submissions[i] = report(i, "Cake (MANUAL: best, x3 0.446->0.453)", X, y, q6)


# ── F7: 6D · ML hyperparam — manual one-var: continue x3 DOWN 0.515 -> 0.503 ─
# x3=0.515 beat 0.525 -> optimum is lower. Pin all else at the new best.
i = 7
X, y = load(i)
best7 = X[np.argmax(y)].copy()  # (0.0051,0.1083,0.5151,0.1939,0.3587,0.7692)
q7 = best7.copy(); q7[2] = 0.5031
submissions[i] = report(i, "ML hyperparam (MANUAL: continue x3 0.515->0.503)", X, y, q7)


# ── F8: 8D · TRUST-REGION GP UCB around the breakthrough best 9.7083 ─────────
i = 8
X, y = load(i)
best8 = X[np.argmax(y)]
kernel8 = C(1.0, (1e-3, 1e3)) * Matern(length_scale=[0.08] * 8, nu=2.5) + WhiteKernel(
    noise_level=1e-3, noise_level_bounds=(1e-6, 0.1))
loo_diagnostic("F8", X, y, kernel8, alpha=1e-5)
gp8 = fit_gp(X, y, kernel8, alpha=1e-5)
# Use the SAME tight half-widths that produced the calibrated W7 breakthrough
# (predicted 9.7073, actual 9.7083), now centred on the new best. A wider box
# let UCB roam to the edges and extrapolate multiple downtrends at once — the
# F6-style failure mode. Tight box = consolidate the breakthrough.
half = np.array([0.012, 0.06, 0.025, 0.05, 0.06, 0.04, 0.05, 0.06])
lo = np.clip(best8 - half, 0.005, 0.995); hi = np.clip(best8 + half, 0.005, 0.995)
rng8 = np.random.default_rng(889)
cand8 = np.column_stack([rng8.uniform(lo[d], hi[d], 120000) for d in range(8)])
acq8, _, _ = ucb_gp(gp8, cand8, kappa=1.5)
q8 = cand8[np.argmax(acq8)]
mu_q8, sig_q8 = gp8.predict(q8.reshape(1, -1), return_std=True)
submissions[i] = report(i, "F8 (trust-region GP UCB κ=1.5, exploit breakthrough)", X, y, q8, float(mu_q8), float(sig_q8))


# ── Final output ───────────────────────────────────────────────────────────────
print("\n\n========== WEEK 8 SUBMISSIONS ==========")
for i in range(1, 9):
    print(f"F{i}: {submissions[i]}")
