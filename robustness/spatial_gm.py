"""
Kelejian-Prucha generalised-moments estimation for spatial error models.

An earlier attempt in this repository minimised a hand-written pair of moments
that were not zero-mean at the truth, because they omitted the trace correction
that the second moment carries. This implements the published three-moment
system instead, in the arrangement used by the reference implementation, and
gates every use of it behind a Monte Carlo check on processes whose parameter is
known.

For u = lam*W*u + e with e independent and homoskedastic, write ubar = W u and
ubarbar = W ubar. The moment conditions are

    E[e'e / n]          = s2
    E[ebar'ebar / n]    = s2 * tr(W'W) / n
    E[ebar'e / n]       = 0

which, substituting e = u - lam*ubar, are linear in (lam, lam^2, s2):

    G [lam, lam^2, s2]' = g

with g and G as below. The estimator minimises the squared residual of that
system subject to the second element being the square of the first.

Reference: Kelejian and Prucha, "A generalized moments estimator for the
autoregressive parameter in a spatial model", International Economic Review 40
(1999). Arrangement follows spreg's _momentsGM_Error.
"""

import numpy as np
from scipy.optimize import minimize


def gm_moments(u, Wu, WWu, trWtW, n):
    g = np.array([u @ u, Wu @ Wu, u @ Wu], dtype=float) / n
    G = np.array([
        [2 * (u @ Wu),        -(Wu @ Wu),    float(n)],
        [2 * (Wu @ WWu),      -(WWu @ WWu),  float(trWtW)],
        [(u @ WWu) + (Wu @ Wu), -(Wu @ WWu), 0.0],
    ], dtype=float) / n
    return G, g


def gm_lambda(u, wlag, trWtW, n=None, bounds=(-0.99, 0.99)):
    """Estimate the spatial error parameter. wlag(v) applies W to a vector."""
    u = np.asarray(u, dtype=float)
    n = len(u) if n is None else n
    Wu = wlag(u)
    WWu = wlag(Wu)
    G, g = gm_moments(u, Wu, WWu, trWtW, n)

    def obj(theta):
        lam, s2 = theta
        r = g - G @ np.array([lam, lam * lam, s2])
        return float(r @ r)

    best, bestval = 0.0, np.inf
    for lam0 in (-0.5, 0.0, 0.5):
        r = minimize(obj, x0=[lam0, float(u @ u) / n],
                     bounds=[bounds, (1e-10, None)], method="L-BFGS-B")
        if r.fun < bestval:
            best, bestval = float(r.x[0]), float(r.fun)
    return best


def _dense_wlag(W):
    return lambda v: W @ v


def monte_carlo(n=200, lam_true=0.6, reps=200, seed=0, kind="cycle"):
    """Recover a known error parameter. This is the gate on the estimator."""
    rng = np.random.default_rng(seed)
    W = np.zeros((n, n))
    if kind == "cycle":
        for i in range(n):
            W[i, (i + 1) % n] = 1.0
            W[i, (i - 1) % n] = 1.0
    else:
        for i in range(n):
            for j in rng.choice([k for k in range(n) if k != i], 4, replace=False):
                W[i, j] = 1.0
    W = W / W.sum(axis=1, keepdims=True)
    A = np.linalg.inv(np.eye(n) - lam_true * W)
    trWtW = np.trace(W.T @ W)
    est = [gm_lambda(A @ rng.normal(size=n), _dense_wlag(W), trWtW, n) for _ in range(reps)]
    return float(np.mean(est)), float(np.std(est))


if __name__ == "__main__":
    print("Monte Carlo check on the estimator (mean estimate over 200 replications)\n")
    ok = True
    for kind in ("cycle", "random"):
        for lam in (-0.4, 0.0, 0.3, 0.6, 0.8):
            m, sd = monte_carlo(lam_true=lam, kind=kind)
            bias = m - lam
            flag = "ok " if abs(bias) < 0.05 else "BIAS"
            ok &= abs(bias) < 0.05
            print(f"  {kind:<7} true {lam:+.2f}   estimate {m:+.4f}  sd {sd:.4f}  bias {bias:+.4f}  {flag}")
    print("\nestimator recovers the known parameter" if ok else "\nESTIMATOR FAILS ITS OWN CHECK")
