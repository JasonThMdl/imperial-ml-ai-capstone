"""Week 7 Bayesian Optimisation queries.

Reads from W6_data. New this week (deferred from W6 handover):
  • LOO calibration diagnostic printed for every GP function (flags |z|>3).
  • F3 uses an INPUT-WARPED GP (Beta-CDF warping, Snoek et al. 2014) to tackle
    the non-stationary cliff (LOO z=9.8 in W5). Warped vs unwarped reported.
  • F8 moved to a TRUST-REGION local GP (TuRBO-style; Eriksson et al. 2019,
    validated by Turner et al. 2021) — replaces the underfit NN ensemble.

Per-function strategy from W6 results:
  F1 — still ≈0; next maximin point (recomputed on current 16 obs)
  F2 — converged (new best 0.6136 at x1≈0.718); confirm in a tiny box
  F3 — W6 overshot (x2/x3 too high); warped GP, pull back, local box
  F4 — new best 0.650; GP PI, tighten to ±0.015
  F5 — line search; gain decelerating (+591), push x1 to 0.98 to test plateau
  F6 — x4=0.80 hurt -> x4 optimum ≈0.77; revert, change ONE lever: lower x2
  F7 — x3=0.535 just missed; razor box around x3=0.525
  F8 — 3rd miss; trust-region local GP UCB around the W3 best (9.622)
"""

import os, sys, warnings
import numpy as np
from scipy.stats import norm, beta as beta_dist
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, RBF, WhiteKernel, ConstantKernel as C

sys.stdout.reconfigure(encoding="utf-8")
warnings.filterwarnings("ignore")

BASE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "data", "W6_data")
RNG = np.random.default_rng(42)


# ── helpers ────────────────────────────────────────────────────────────────────

def load(i):
    X = np.load(os.path.join(BASE, f"function_{i}", f"f{i}_initial_inputs.npy"))
    y = np.load(os.path.join(BASE, f"function_{i}", f"f{i}_initial_outputs.npy"))
    return X, y


def fit_gp(X, y, kernel, alpha=1e-6, n_restarts=10):
    gp = GaussianProcessRegressor(kernel=kernel, alpha=alpha, normalize_y=True,
                                  n_restarts_optimizer=n_restarts, random_state=0)
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
    mu, sigma = gp.predict(X, return_std=True)
    with np.errstate(divide="ignore", invalid="ignore"):
        z = (mu - y_best - xi) / sigma
        out = norm.cdf(z)
        out[sigma < 1e-12] = 0.0
    return out, mu, sigma


def ucb_gp(gp, X, kappa):
    mu, sigma = gp.predict(X, return_std=True)
    return mu + kappa * sigma, mu, sigma


def loo_diagnostic(tag, X, y, kernel, alpha=1e-6):
    """Leave-one-out standardized residuals. A well-specified stationary GP
    gives z~N(0,1). Flags localized misfit (|z|>3 = unreliable uncertainty)."""
    n = len(y); z = np.zeros(n)
    for k in range(n):
        m = np.ones(n, bool); m[k] = False
        gp = GaussianProcessRegressor(kernel=kernel, alpha=alpha, normalize_y=True,
                                      n_restarts_optimizer=3, random_state=0)
        gp.fit(X[m], y[m])
        mu, sd = gp.predict(X[k:k+1], return_std=True)
        z[k] = (y[k] - mu[0]) / sd[0]
    flag = "  ⚠ |z|>3 — uncertainty NOT trustworthy" if np.abs(z).max() > 3 else "  ✓ calibrated"
    print(f"   [LOO {tag}] std(z)={z.std():.2f}  max|z|={np.abs(z).max():.1f}{flag}")
    return z.std(), float(np.abs(z).max())


def correlations(X, y):
    return [round(float(np.corrcoef(X[:, d], y)[0, 1]), 3) for d in range(X.shape[1])]


def report(i, name, X, y, q, mu_q=None, sigma_q=None):
    print(f"\n=== F{i}  {name}  (n={len(y)}, d={X.shape[1]}) ===")
    print(f"  Best y so far : {y.max():.6f} at {[f'{v:.4f}' for v in X[np.argmax(y)]]}")
    print(f"  Last y (W6)   : {y[-1]:.6f}")
    print(f"  Correlations  : {correlations(X, y)}")
    print(f"  Proposed query: {[f'{v:.6f}' for v in q]}")
    if mu_q is not None:
        print(f"  Surrogate at q: mu={mu_q:.4f}  sigma={sigma_q:.4f}")
    return "-".join(f"{v:.6f}" for v in q)


submissions = {}


# ── F1: 2D · Maximin (recomputed on current 16 obs) ──────────────────────────
i = 1
X, y = load(i)
cand = RNG.uniform(0.005, 0.995, size=(80000, 2))
mins = np.min(np.linalg.norm(cand[:, None, :] - X[None, :, :], axis=2), axis=1)
q1 = cand[np.argmax(mins)]
submissions[i] = report(i, "Radiation (maximin, largest current gap)", X, y, q1)


# ── F2: 2D · Noisy ML — converged; confirm in tiny box ───────────────────────
# New best 0.6136 at (0.718, 0.968). x1=0.714→0.571, 0.715→0.598, 0.718→0.614.
# Essentially converged. Tiny box to confirm / squeeze.
i = 2
X, y = load(i)
kernel2 = C(1.0, (1e-3, 1e3)) * Matern(length_scale=[0.15, 0.05], nu=2.5) + WhiteKernel(
    noise_level=1e-3, noise_level_bounds=(1e-6, 0.1))
loo_diagnostic("F2", X, y, kernel2)
gp = fit_gp(X, y, kernel2)
rng2 = np.random.default_rng(222)
cand = np.column_stack([rng2.uniform(0.714, 0.722, 60000), rng2.uniform(0.963, 0.974, 60000)])
acq, mu, sigma = ei_gp(gp, cand, y.max())
q2 = cand[np.argmax(acq)]
mu_q2, sig_q2 = gp.predict(q2.reshape(1, -1), return_std=True)
submissions[i] = report(i, "Noisy ML (GP EI, converged box)", X, y, q2, float(mu_q2), float(sig_q2))


# ── F3: 3D · Drug — INPUT-WARPED GP (Beta-CDF), local box ────────────────────
# W6 overshot: pushing x2→0.543, x3→0.417 gave -0.0119 < best -0.0077 at
# (0.466, 0.517, 0.426). LOO in W5 showed z=9.8 (non-stationary cliff).
# Beta-CDF input warping (Snoek et al. 2014): warp each dim x -> I_x(a,b) and
# pick (a,b) per dim by maximising GP marginal likelihood (coordinate descent).
i = 3
X, y = load(i)
base_kernel3 = C(1.0, (1e-3, 1e3)) * Matern(length_scale=[0.15, 0.15, 0.10], nu=0.5)

def warp(Xin, ab):
    Xc = np.clip(Xin, 1e-6, 1 - 1e-6)
    return np.column_stack([beta_dist.cdf(Xc[:, d], ab[d][0], ab[d][1]) for d in range(Xc.shape[1])])

# unwarped LOO baseline
print()
loo_diagnostic("F3 unwarped", X, y, base_kernel3)

# coordinate-descent over Beta (a,b) per dimension, maximising LML
grid = [0.5, 1.0, 1.5, 2.0, 3.0]
ab = [(1.0, 1.0)] * 3
def lml(ab_try):
    gpw = GaussianProcessRegressor(kernel=base_kernel3, alpha=1e-6, normalize_y=True,
                                   n_restarts_optimizer=2, random_state=0)
    gpw.fit(warp(X, ab_try), y)
    return gpw.log_marginal_likelihood(gpw.kernel_.theta)
for _ in range(2):
    for d in range(3):
        best_ab, best_l = ab[d], -1e18
        for a in grid:
            for b in grid:
                trial = ab.copy(); trial[d] = (a, b)
                l = lml(trial)
                if l > best_l:
                    best_l, best_ab = l, (a, b)
        ab[d] = best_ab
print(f"   [F3 warp] selected Beta(a,b) per dim: {[ (round(a,1),round(b,1)) for a,b in ab ]}")

# LOO on warped inputs
Xw = warp(X, ab)
loo_diagnostic("F3 warped  ", Xw, y, base_kernel3)

# Warping reduced the cliff (z: 10.0->8.6) but did NOT make the GP trustworthy
# (max|z| still 8.6), and its UCB pointed back at x2≈0.54 — the same overshoot
# that failed in W6 (0.543 -> -0.0119). So we do NOT trust the surrogate here.
# MANUAL clean experiment instead: take the known best (0.466, 0.517, 0.426)
# and move ONLY x2 downward (0.517 -> 0.49) — the opposite of the W6 failure,
# isolating the x2 gradient near the peak.
best3 = X[np.argmax(y)].copy()
q3 = best3.copy()
q3[1] = 0.49
submissions[i] = report(i, "Drug (MANUAL local: best, x2 0.517->0.49)", X, y, q3)


# ── F4: 4D · Warehouse — GP PI, tighten to ±0.015 ────────────────────────────
# New best 0.650 at (0.404, 0.410, 0.361, 0.426). PI compounding: 0.524→0.566→0.650.
i = 4
X, y = load(i)
best4 = X[np.argmax(y)]
kernel4 = C(1.0, (1e-3, 1e3)) * Matern(length_scale=[0.1] * 4, nu=1.5) + WhiteKernel(
    noise_level=0.1, noise_level_bounds=(1e-4, 1.0))
loo_diagnostic("F4", X, y, kernel4, alpha=1e-4)
gp4 = fit_gp(X, y, kernel4, alpha=1e-4)
rng4 = np.random.default_rng(444)
bounds4 = [(max(0.005, best4[d] - 0.015), min(0.995, best4[d] + 0.015)) for d in range(4)]
cand4 = np.column_stack([rng4.uniform(b[0], b[1], 80000) for b in bounds4])
acq4, mu4, sigma4 = pi_gp(gp4, cand4, y.max(), xi=0.005)
q4 = cand4[np.argmax(acq4)]
mu_q4, sig_q4 = gp4.predict(q4.reshape(1, -1), return_std=True)
submissions[i] = report(i, "Warehouse (GP PI ±0.015 around new best)", X, y, q4, float(mu_q4), float(sig_q4))


# ── F5: 4D · Chemical — line search, push x1 to 0.98 (test plateau) ──────────
# Gains decelerating: +1052, +1446, then +591. Push x1 to 0.98 to confirm.
i = 5
X, y = load(i)
mask = (X[:, 1] >= 0.985) & (X[:, 2] >= 0.985) & (X[:, 3] >= 0.985)
x1_c, y_c = X[mask, 0], y[mask]
coeffs = np.polyfit(x1_c, y_c, deg=2)
bc = mask.nonzero()[0][np.argmax(y_c)]
x1_next = min(0.980, 0.995)
q5 = np.array([x1_next, float(X[bc, 1]), float(X[bc, 2]), float(X[bc, 3])])
pred = float(np.polyval(coeffs, x1_next))
print(f"\n[F5] ceiling x1={np.round(x1_c,3)} y={np.round(y_c,0)}  poly pred@0.98≈{pred:.0f}")
submissions[i] = report(i, "Chemical (line search, x1→0.98 plateau test)", X, y, q5)


# ── F6: 5D · Cake — MANUAL one-variable: lower x2 ────────────────────────────
# W6: x4 0.770→0.80 dropped to -0.462, so x4 optimum ≈0.77 (= best). Revert.
# x2 is the strongest untested lever (r=-0.43); all top points sit at x2≈0.34.
# Take the best, change ONLY x2 (0.349 → 0.27) to test the negative signal.
i = 6
X, y = load(i)
best6 = X[np.argmax(y)].copy()  # (0.325, 0.349, 0.446, 0.770, 0.010)
q6 = best6.copy()
q6[1] = 0.27
submissions[i] = report(i, "Cake (MANUAL: best, x2 0.349->0.27)", X, y, q6)


# ── F7: 6D · ML hyperparam — GP EI, razor box around x3=0.525 ────────────────
# W6: x3=0.535 → 2.267, just below best 2.274 at x3=0.525. Tighten x3 hard.
i = 7
X, y = load(i)
kernel7 = C(1.0, (1e-3, 1e3)) * Matern(length_scale=[0.05] * 6, nu=1.5) + WhiteKernel(
    noise_level=1e-3, noise_level_bounds=(1e-6, 0.1))
loo_diagnostic("F7", X, y, kernel7, alpha=1e-5)
gp7 = fit_gp(X, y, kernel7, alpha=1e-5)
rng7 = np.random.default_rng(777)
bounds7 = [(0.005, 0.010), (0.090, 0.112), (0.515, 0.532), (0.170, 0.195),
           (0.345, 0.385), (0.760, 0.785)]
cand7 = np.column_stack([rng7.uniform(b[0], b[1], 60000) for b in bounds7])
acq7, mu7, sigma7 = ei_gp(gp7, cand7, y.max())
q7 = cand7[np.argmax(acq7)]
mu_q7, sig_q7 = gp7.predict(q7.reshape(1, -1), return_std=True)
submissions[i] = report(i, "ML hyperparam (GP EI, x3∈[0.515,0.532])", X, y, q7, float(mu_q7), float(sig_q7))


# ── F8: 8D · TRUST-REGION local GP UCB around the W3 best (9.622) ────────────
# Three straight misses with the NN ensemble (which is underfit). Switch to a
# local GP in a tight box around the best (TuRBO-style). Strongest signals:
# x1,x3 r≈-0.73 (low, but NOT floor — best 0.027/0.080), x7 r=-0.51 (low).
i = 8
X, y = load(i)
best8 = X[np.argmax(y)]
kernel8 = C(1.0, (1e-3, 1e3)) * Matern(length_scale=[0.08] * 8, nu=2.5) + WhiteKernel(
    noise_level=1e-3, noise_level_bounds=(1e-6, 0.1))
loo_diagnostic("F8", X, y, kernel8, alpha=1e-5)
gp8 = fit_gp(X, y, kernel8, alpha=1e-5)
# tight trust region around the best (asymmetric: allow strong-signal dims to go lower)
half = np.array([0.012, 0.06, 0.025, 0.05, 0.06, 0.04, 0.05, 0.06])
lo = np.clip(best8 - half, 0.005, 0.995)
hi = np.clip(best8 + half, 0.005, 0.995)
rng8 = np.random.default_rng(888)
cand8 = np.column_stack([rng8.uniform(lo[d], hi[d], 100000) for d in range(8)])
acq8, mu8, sigma8 = ucb_gp(gp8, cand8, kappa=1.5)
q8 = cand8[np.argmax(acq8)]
mu_q8, sig_q8 = gp8.predict(q8.reshape(1, -1), return_std=True)
submissions[i] = report(i, "F8 (trust-region local GP UCB κ=1.5)", X, y, q8, float(mu_q8), float(sig_q8))


# ── Final output ───────────────────────────────────────────────────────────────
print("\n\n========== WEEK 7 SUBMISSIONS ==========")
for i in range(1, 9):
    print(f"F{i}: {submissions[i]}")
