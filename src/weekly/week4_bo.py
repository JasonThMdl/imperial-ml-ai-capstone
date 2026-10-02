"""Week 4 Bayesian Optimisation queries — MLP Ensemble surrogate.

Switches to a bootstrap MLP ensemble for uncertainty estimation, replacing
the GP for higher-dimensional functions. The ensemble mean/std feed a UCB
acquisition; scipy L-BFGS-B then refines the best candidate using the
neural network's implicit gradient signal (finite-difference backprop).

Strategies are informed by W3 outcomes:
  F1 — still flat; continue space-filling (maximin)
  F2 — exploit near best (0.721, 0.970); GP + EI, tight box
  F3 — new best -0.0287; exploit tighter around (0.470, 0.483, 0.401)
  F4 — W3 catastrophic (-2.59); pivot to very tight exploitation of best
  F5 — big jump to 4486; keep pushing x1 higher, x2-x4 near ceiling
  F6 — stuck at -0.357; x5 must be LOW (r=-0.66), x4 HIGH (r=+0.57)
  F7 — stuck at 2.274; x3 MUST be ~0.50 (W3 dropped it to 0.27 → bad)
  F8 — improving; constrained NN search with all W3 learnings
"""

import os, sys, warnings
import numpy as np
from scipy.stats import norm
from scipy.optimize import minimize
from sklearn.neural_network import MLPRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, RBF, WhiteKernel, ConstantKernel as C

sys.stdout.reconfigure(encoding="utf-8")
warnings.filterwarnings("ignore")

BASE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "data", "W3_data")
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


def ucb_gp(gp, X, kappa):
    mu, sigma = gp.predict(X, return_std=True)
    return mu + kappa * sigma, mu, sigma


def build_ensemble(X_train, y_train, n_models=15, seed=0):
    """Bootstrap MLP ensemble (64-64 ReLU).  Returns list of fitted models."""
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


def ensemble_predict(models, X):
    preds = np.array([m.predict(X) for m in models])
    return preds.mean(axis=0), preds.std(axis=0)


def nn_ucb(models, X, kappa=2.0):
    mu, sigma = ensemble_predict(models, X)
    return mu + kappa * sigma, mu, sigma


def nn_ei(models, X, y_best, xi=0.0):
    mu, sigma = ensemble_predict(models, X)
    with np.errstate(divide="ignore", invalid="ignore"):
        imp = mu - y_best - xi
        z = np.where(sigma > 1e-12, imp / sigma, 0.0)
        out = np.where(sigma > 1e-12, imp * norm.cdf(z) + sigma * norm.pdf(z), 0.0)
    return out, mu, sigma


def nn_best(X_train, y_train, bounds, kappa=2.0, n_cand=80000,
            use_ei=False, seed=0, n_models=15):
    """Fit ensemble, UCB/EI over random candidates, refine with L-BFGS-B."""
    models = build_ensemble(X_train, y_train, n_models=n_models, seed=seed)
    d = X_train.shape[1]
    rng = np.random.default_rng(seed + 500)
    cand = np.column_stack([rng.uniform(b[0], b[1], n_cand) for b in bounds])

    if use_ei:
        acq, mu, sigma = nn_ei(models, cand, y_train.max())
    else:
        acq, mu, sigma = nn_ucb(models, cand, kappa)

    best_cand = cand[np.argmax(acq)]

    # Gradient-guided refinement: L-BFGS-B with finite-difference gradients
    # through the NN predictions (exploits backprop-like signal).
    def neg_acq(x):
        xr = x.reshape(1, -1)
        ps = np.array([m.predict(xr)[0] for m in models])
        if use_ei:
            mu_x, sig_x = ps.mean(), ps.std()
            imp = mu_x - y_train.max()
            z = imp / sig_x if sig_x > 1e-12 else 0.0
            val = imp * norm.cdf(z) + sig_x * norm.pdf(z) if sig_x > 1e-12 else 0.0
        else:
            val = ps.mean() + kappa * ps.std()
        return -val

    result = minimize(neg_acq, best_cand, method="L-BFGS-B", bounds=bounds,
                      options={"maxiter": 300, "ftol": 1e-9})
    q = result.x if result.success else best_cand

    preds_q = np.array([m.predict(q.reshape(1, -1))[0] for m in models])
    return q, preds_q.mean(), preds_q.std()


def correlations(X, y):
    return [round(float(np.corrcoef(X[:, d], y)[0, 1]), 3) for d in range(X.shape[1])]


def report(i, name, X, y, q, mu_q, sigma_q):
    top_idx = np.argsort(y)[-5:][::-1]
    print(f"\n=== F{i}  {name}  (n={len(y)}, d={X.shape[1]}) ===")
    print(f"  Best y so far : {y.max():.4f} at {[f'{v:.4f}' for v in X[np.argmax(y)]]}")
    print(f"  Last y (W3)   : {y[-1]:.4f}")
    print(f"  Correlations  : {correlations(X, y)}")
    print(f"  Proposed query: {[f'{v:.6f}' for v in q]}")
    print(f"  Surrogate at q: mu={mu_q:.4f}  sigma={sigma_q:.4f}")
    return "-".join(f"{v:.6f}" for v in q)


submissions = {}


# ── F1: 2D · Flat (radiation) — maximin space-filling ─────────────────────────
i = 1
X, y = load(i)
# All 13 queries returned 0; GP is degenerate. Continue space-filling by
# finding the point in [0,1]^2 that maximises its minimum distance to all
# existing observations.
cand = RNG.uniform(0.005, 0.995, size=(30000, 2))
mins = np.min(np.linalg.norm(cand[:, None, :] - X[None, :, :], axis=2), axis=1)
q1 = cand[np.argmax(mins)]
mu_q1, sigma_q1 = 0.0, 0.0
submissions[i] = report(i, "Radiation (FLAT, maximin)", X, y, q1, mu_q1, sigma_q1)


# ── F2: 2D · Noisy ML — GP EI, tight box around best ─────────────────────────
# Best: (0.721, 0.970). W3 query lowered x2 to 0.867 → dropped to 0.549.
# Lesson: x2 MUST stay ≥ 0.945. Exploit narrow band around x1≈0.72.
i = 2
X, y = load(i)
kernel = C(1.0, (1e-3, 1e3)) * Matern(length_scale=[0.15, 0.05], nu=2.5) + WhiteKernel(
    noise_level=1e-3, noise_level_bounds=(1e-6, 0.1)
)
gp = fit_gp(X, y, kernel)
rng2 = np.random.default_rng(22)
cand = np.column_stack([
    rng2.uniform(0.67, 0.76, 60000),    # tight around x1≈0.72
    rng2.uniform(0.945, 0.990, 60000),  # x2 must stay high
])
acq, mu, sigma = ei_gp(gp, cand, y.max())
q2 = cand[np.argmax(acq)]
mu_q2, sig_q2 = gp.predict(q2.reshape(1, -1), return_std=True)
submissions[i] = report(i, "Noisy ML (GP EI, x2≥0.945)", X, y, q2,
                        float(mu_q2), float(sig_q2))


# ── F3: 3D · Non-smooth drug — GP UCB, exploit new best ──────────────────────
# New best -0.0287 at (0.470, 0.483, 0.401). x3 correlation is -0.55 (overall
# negative), but best two points both have x3 in [0.34, 0.40]. Cap x3 at 0.44
# to avoid pushing into unseen territory; x3=0.50 was hitting the upper bound.
i = 3
X, y = load(i)
kernel = C(1.0, (1e-3, 1e3)) * Matern(length_scale=[0.15, 0.15, 0.10], nu=0.5)
gp = fit_gp(X, y, kernel)
rng3 = np.random.default_rng(33)
cand = np.column_stack([
    rng3.uniform(0.37, 0.57, 60000),   # around x1≈0.47
    rng3.uniform(0.35, 0.60, 60000),   # around x2≈0.48
    rng3.uniform(0.33, 0.44, 60000),   # x3 capped at 0.44 (best was 0.40)
])
acq, mu, sigma = ucb_gp(gp, cand, kappa=2.0)
q3 = cand[np.argmax(acq)]
mu_q3, sig_q3 = gp.predict(q3.reshape(1, -1), return_std=True)
submissions[i] = report(i, "Drug (GP UCB κ=2.0, exploit new best)", X, y, q3,
                        float(mu_q3), float(sig_q3))


# ── F4: 4D · Multimodal warehouse — GP EI, tight ±0.07 around only good point ─
# Only 1 point in history with y>0 (best=0.5235 at (0.408,0.411,0.347,0.433)).
# W3 UCB sent us to a catastrophic region (-2.59). NN ensemble was wildly
# uncertain (sigma=6.6) with only 33 points. GP is more reliable here:
# better inductive bias for the smooth neighbourhood around a single peak.
i = 4
X, y = load(i)
best_x4 = X[np.argmax(y)]
kernel4 = C(1.0, (1e-3, 1e3)) * Matern(length_scale=[0.1] * 4, nu=1.5) + WhiteKernel(
    noise_level=0.1, noise_level_bounds=(1e-4, 1.0)
)
gp4 = fit_gp(X, y, kernel4, alpha=1e-4, n_restarts=10)
rng4 = np.random.default_rng(44)
bounds4 = [(max(0.005, best_x4[d] - 0.07), min(0.995, best_x4[d] + 0.07))
           for d in range(4)]
cand4 = np.column_stack([rng4.uniform(b[0], b[1], 60000) for b in bounds4])
acq4, mu4, sigma4 = ei_gp(gp4, cand4, y.max())
q4 = cand4[np.argmax(acq4)]
mu_q4, sig_q4 = gp4.predict(q4.reshape(1, -1), return_std=True)
submissions[i] = report(i, "Warehouse (GP EI, tight ±0.07 around best)", X, y, q4,
                        float(mu_q4), float(sig_q4))


# ── F5: 4D · Chemical — GP EI, push x1 higher ────────────────────────────────
# Trajectory: x1=0.265→4257, x1=0.304→4270, x1=0.544→4486 (new best).
# Strong signal: higher x1 → higher y. Push x1 into [0.55, 0.78].
# x2/x3/x4 stay pinned at ceiling (≥0.985).
i = 5
X, y = load(i)
kernel = C(1.0, (1e-3, 1e3)) * RBF(length_scale=[0.15] * 4)
gp = fit_gp(X, y, kernel)
rng5 = np.random.default_rng(55)
cand = np.column_stack([
    rng5.uniform(0.55, 0.78, 80000),
    rng5.uniform(0.985, 0.995, 80000),
    rng5.uniform(0.985, 0.995, 80000),
    rng5.uniform(0.985, 0.995, 80000),
])
acq, mu, sigma = ei_gp(gp, cand, y.max())
q5 = cand[np.argmax(acq)]
mu_q5, sig_q5 = gp.predict(q5.reshape(1, -1), return_std=True)
submissions[i] = report(i, "Chemical (GP EI, x1∈[0.55,0.78])", X, y, q5,
                        float(mu_q5), float(sig_q5))


# ── F6: 5D · Cake — NN UCB, x5 very low, x4 high ────────────────────────────
# Best: (0.361, 0.338, 0.456, 0.746, 0.056). Correlations: x5 r=-0.66 (must
# be near 0), x4 r=+0.57 (must be high). W3 dropped x3 and raised x4/x5 →
# worse. Fix: pin x5 ≤ 0.08, keep x4 ≥ 0.70, x3 ∈ [0.38, 0.55].
i = 6
X, y = load(i)
bounds6 = [
    (0.22, 0.50),    # x1 around 0.36
    (0.18, 0.50),    # x2 around 0.34 (r=-0.39, keep low-ish)
    (0.38, 0.55),    # x3 ≈ 0.46 (r=+0.19)
    (0.68, 0.85),    # x4 high (r=+0.57)
    (0.005, 0.08),   # x5 very low (r=-0.66)
]
q6, mu_q6, sig_q6 = nn_best(X, y, bounds6, kappa=2.0, seed=66)
submissions[i] = report(i, "Cake (NN UCB κ=2.0, x5≤0.08, x4≥0.68)", X, y, q6,
                        mu_q6, sig_q6)


# ── F7: 6D · ML hyperparam — NN UCB, x3 pinned to ~0.50 ─────────────────────
# Best: (0.005, 0.101, 0.525, 0.182, 0.365, 0.771). W3 query had x3=0.272
# (too low) → result 1.37 vs 2.27. Critical fix: keep x3 ∈ [0.45, 0.65].
# x1 must be near 0 (r=-0.53), x6 should be high (r=+0.43).
i = 7
X, y = load(i)
bounds7 = [
    (0.005, 0.025),  # x1 near 0 (r=-0.53)
    (0.07, 0.16),    # x2 low (r=-0.24)
    (0.45, 0.65),    # x3 ≈ 0.50 — critical constraint
    (0.13, 0.25),    # x4 moderate-low (r=-0.31)
    (0.30, 0.45),    # x5 moderate (r=-0.28)
    (0.72, 0.88),    # x6 high (r=+0.43)
]
q7, mu_q7, sig_q7 = nn_best(X, y, bounds7, kappa=2.0, seed=77)
submissions[i] = report(i, "ML hyperparam (NN UCB κ=2.0, x3∈[0.45,0.65])", X, y, q7,
                        mu_q7, sig_q7)


# ── F8: 8D · NN UCB, tighten all constraints from W3 analysis ────────────────
# Best: (0.027,0.291,0.080,0.181,0.290,0.940,0.126,0.804).
# Strong correlations: x1 r=-0.69 (very low), x3 r=-0.69 (very low),
# x7 r=-0.43 (low), x6 r=+0.22 (high), x8 r=+0.18 (high).
# Tighten x1 and x3 further (<0.06) to push harder on the strongest signals.
i = 8
X, y = load(i)
bounds8 = [
    (0.005, 0.06),   # x1 very low
    (0.05, 0.35),    # x2 moderate-low
    (0.005, 0.06),   # x3 very low
    (0.05, 0.25),    # x4 lower
    (0.20, 0.40),    # x5 moderate
    (0.88, 0.99),    # x6 high
    (0.05, 0.18),    # x7 low (r=-0.43)
    (0.78, 0.96),    # x8 high
]
q8, mu_q8, sig_q8 = nn_best(X, y, bounds8, kappa=2.0, seed=88)
submissions[i] = report(i, "F8 (NN UCB κ=2.0, x1/x3<0.06, x6/x8 high)", X, y, q8,
                        mu_q8, sig_q8)


# ── Final output ───────────────────────────────────────────────────────────────
print("\n\n========== WEEK 4 SUBMISSIONS ==========")
for i in range(1, 9):
    print(f"F{i}: {submissions[i]}")
