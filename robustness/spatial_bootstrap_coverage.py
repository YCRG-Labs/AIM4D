"""
Does the country-block bootstrap actually cover?

The spatial intervals in this paper come from resampling whole countries with
replacement and repeating the estimation under each draw's own network. That is
a choice about which dependence to preserve, and a point estimate plus a
percentile interval is not evidence that the interval covers. This checks it on
simulated panels built to match the features of the real one that could break it:
a spatial error process, an unbalanced panel, countries that drop out, and rows
with no observed neighbour.

Reported for each design is the share of replications whose 95 per cent interval
contains the true error parameter. A procedure that covers at roughly the nominal
rate can be reported as an interval; one that does not, cannot.

Outputs robustness/spatial_bootstrap_coverage.csv.
"""

import os
import sys
import warnings

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from spatial_gm import gm_lambda
from spatial_models import YearBlockW, ols

OUT = os.path.dirname(os.path.abspath(__file__))
N_REP = int(os.environ.get("AIM4D_COVER_REPS", "200"))
N_BOOT = int(os.environ.get("AIM4D_COVER_BOOT", "120"))


def make_W(n_country, seed, degree=3):
    rng = np.random.default_rng(seed)
    W = np.zeros((n_country, n_country))
    for i in range(n_country):
        for j in rng.choice([k for k in range(n_country) if k != i], degree, replace=False):
            W[i, j] = 1.0
            W[j, i] = 1.0
    return W


def simulate(n_country, n_year, lam, W, seed, missing=0.0):
    """Panel with a spatial error process inside each year, optionally unbalanced."""
    rng = np.random.default_rng(seed)
    cid, yr, yv, xv = [], [], [], []
    order = list(range(n_country))
    for t in range(n_year):
        keep = np.array([i for i in order if rng.random() >= missing])
        if len(keep) < 5:
            continue
        sub = W[np.ix_(keep, keep)]
        rs = sub.sum(axis=1, keepdims=True)
        sub = np.divide(sub, rs, out=np.zeros_like(sub), where=rs > 0)
        e = rng.normal(size=len(keep))
        u = np.linalg.solve(np.eye(len(keep)) - lam * sub, e)
        x = rng.normal(size=len(keep))
        cid.extend(keep.tolist()); yr.extend([t] * len(keep))
        xv.extend(x.tolist()); yv.extend((0.5 * x + u).tolist())
    return (np.array(cid), np.array(yr), np.array(yv), np.array(xv))


def lam_hat(cid, yr, y, x, W, order):
    Wop = YearBlockW(cid, yr, W, order)
    X = np.column_stack([np.ones(len(y)), x])
    _, u = ols(y, X)
    return gm_lambda(u, Wop, Wop.trWtW, Wop.n), Wop


def one_rep(n_country, n_year, lam, W, order, seed, missing, n_boot):
    cid, yr, y, x = simulate(n_country, n_year, lam, W, seed, missing)
    point, _ = lam_hat(cid, yr, y, x, W, order)
    if not np.isfinite(point):
        return None
    rng = np.random.default_rng(seed + 99991)
    uniq = np.unique(cid)
    idx = {c: np.where(cid == c)[0] for c in uniq}
    draws = []
    for _ in range(n_boot):
        pick = rng.choice(uniq, len(uniq), replace=True)
        j = np.concatenate([idx[c] for c in pick])
        v, _ = lam_hat(cid[j], yr[j], y[j], x[j], W, order)
        if np.isfinite(v):
            draws.append(v)
    if len(draws) < n_boot // 2:
        return None
    lo, hi = np.percentile(draws, [2.5, 97.5])
    return point, lo, hi, lo <= lam <= hi



def residual_bootstrap_ci(cid, yr, y, x, W, order, n_boot, seed, wild=True):
    """Jin-Lee style residual bootstrap: hold the network fixed, regenerate the data.

    Resampling countries destroys the dependence the parameter measures, which is
    why it fails to cover. Here the weight matrix and the sample are held exactly
    as observed; only the innovations are resampled, and the outcome is rebuilt
    through the estimated error process, so every bootstrap sample has the same
    network as the data.
    """
    Wop = YearBlockW(cid, yr, W, order)
    X = np.column_stack([np.ones(len(y)), x])
    beta, u = ols(y, X)
    lam = gm_lambda(u, Wop, Wop.trWtW, Wop.n)
    if not np.isfinite(lam):
        return None
    eps = u - lam * Wop(u)                      # innovations implied by the fit
    eps = eps - eps.mean()
    rng = np.random.default_rng(seed)
    draws = []
    for _ in range(n_boot):
        if wild:                                 # Rademacher weights
            e = eps * rng.choice([-1.0, 1.0], size=len(eps))
        else:
            e = rng.choice(eps, size=len(eps), replace=True)
        u_star = e.copy()                        # (I - lam W)^-1 e by Neumann
        term = e.copy()
        for _ in range(40):
            term = lam * Wop(term)
            u_star = u_star + term
            if np.max(np.abs(term)) < 1e-10:
                break
        y_star = X @ beta + u_star
        _, us = ols(y_star, X)
        v = gm_lambda(us, Wop, Wop.trWtW, Wop.n)
        if np.isfinite(v):
            draws.append(v)
    if len(draws) < n_boot // 2:
        return None
    draws = np.array(draws)
    # percentile interval centred by the bootstrap's own bias
    lo, hi = np.percentile(draws, [2.5, 97.5])
    return lam, lo, hi, (lo <= 0) or True, draws.mean()


def one_rep_resid(n_country, n_year, lam, W, order, seed, missing, n_boot):
    cid, yr, y, x = simulate(n_country, n_year, lam, W, seed, missing)
    r = residual_bootstrap_ci(cid, yr, y, x, W, order, n_boot, seed + 4242)
    if r is None:
        return None
    point, lo, hi, _, bmean = r
    return point, lo, hi, bool(lo <= lam <= hi)



def one_rep_rho(n_country, n_year, rho, W, order, seed, missing, n_boot):
    """Coverage for the autoregressive parameter, not just the error parameter.

    The earlier study checked lambda only. An interval procedure validated for one
    parameter is not validated for another, and the rho intervals were generated by
    a bootstrap whose data-generating process had no rho in it.
    """
    rng = np.random.default_rng(seed)
    cid, yr, y0, x = simulate(n_country, n_year, 0.0, W, seed, missing)
    Wop = YearBlockW(cid, yr, W, order)

    def neumann(v, c, it=40):
        acc, t = v.copy(), v.copy()
        for _ in range(it):
            t = c * Wop(t); acc = acc + t
            if np.max(np.abs(t)) < 1e-10: break
        return acc

    X = np.column_stack([np.ones(len(y0)), x])
    y = neumann(X @ np.array([0.0, 0.5]) + rng.normal(size=len(y0)), rho)
    Wy = Wop(y)
    inst = np.column_stack([Wop(x), Wop(Wop(x))])
    from spatial_models import tsls
    b, _, _, _ = tsls(y, Wy[:, None], X, inst, cid)
    point = float(b[0])
    rho0 = float(np.clip(point, -0.9, 0.9))
    beta0, _ = ols(y - rho0 * Wy, X)
    u = y - rho0 * Wy - X @ beta0
    u = u - u.mean()
    draws = []
    for _ in range(n_boot):
        e = u * rng.choice([-1.0, 1.0], size=len(u))
        ys = neumann(X @ beta0 + e, rho0)
        bb, _, _, _ = tsls(ys, Wop(ys)[:, None], X, inst, cid)
        draws.append(float(bb[0]))
    lo, hi = np.percentile(draws, [2.5, 97.5])
    return point, lo, hi, bool(lo <= rho <= hi)



def one_rep_lee(n_country, n_year, rho, W, order, seed, missing, n_boot):
    """Coverage for the ITERATED best-instrument estimator specifically.

    The rho study above validates the interval procedure for the Kelejian-Prucha
    instrument set. That does not validate it for a different estimator, and the
    paper's only interval excluding zero comes from the iterated best instrument,
    so it needs its own check.
    """
    from spatial_models import tsls, first_stage_F
    rng = np.random.default_rng(seed)
    cid, yr, y0, x = simulate(n_country, n_year, 0.0, W, seed, missing)
    Wop = YearBlockW(cid, yr, W, order)

    def neumann(v, c, it=40):
        acc, t = v.copy(), v.copy()
        for _ in range(it):
            t = c * Wop(t); acc = acc + t
            if np.max(np.abs(t)) < 1e-12: break
        return acc

    X = np.column_stack([np.ones(len(y0)), x])
    y = neumann(X @ np.array([0.0, 0.5]) + rng.normal(size=len(y0)), rho)

    def iterated_lee(yv):
        Wy = Wop(yv)
        inst = np.column_stack([Wop(x), Wop(Wop(x))])
        r0 = float(np.clip(tsls(yv, Wy[:, None], X, inst, cid)[0][0], -0.95, 0.95))
        for _ in range(40):
            b0, _ = ols(yv - r0 * Wy, X)
            z = Wop(neumann(X @ b0, r0))[:, None]
            nxt = float(np.clip(tsls(yv, Wy[:, None], X, z, cid)[0][0], -0.95, 0.95))
            if abs(nxt - r0) < 1e-6:
                r0 = nxt; break
            r0 = nxt
        return r0

    point = iterated_lee(y)
    Wy = Wop(y)
    beta0, _ = ols(y - point * Wy, X)
    u = y - point * Wy - X @ beta0
    u = u - u.mean()
    draws = []
    for _ in range(n_boot):
        e = u * rng.choice([-1.0, 1.0], size=len(u))
        ys = neumann(X @ beta0 + e, point)
        draws.append(iterated_lee(ys))
    lo, hi = np.percentile(draws, [2.5, 97.5])
    return point, lo, hi, bool(lo <= rho <= hi)


def main():
    rows = []
    designs = [("balanced", 60, 25, 0.0), ("unbalanced 15% missing", 60, 25, 0.15),
               ("unbalanced 30% missing", 60, 25, 0.30)]
    for label, nc, ny, miss in designs:
        for lam in (0.0, 0.3, 0.6):
            W = make_W(nc, seed=11)
            order = list(range(nc))
            kind_env = os.environ.get("AIM4D_BOOT_KIND", "country")
            fn = {"residual": one_rep_resid, "rho": one_rep_rho, "lee": one_rep_lee}.get(kind_env, one_rep)
            res = [fn(nc, ny, lam, W, order, s, miss, N_BOOT) for s in range(N_REP)]
            res = [r for r in res if r]
            if not res:
                continue
            pts = np.array([r[0] for r in res])
            cov = float(np.mean([r[3] for r in res]))
            width = float(np.mean([r[2] - r[1] for r in res]))
            rows.append({"design": label, "lambda_true": lam, "n_rep": len(res),
                         "mean_estimate": round(float(pts.mean()), 4),
                         "bias": round(float(pts.mean() - lam), 4),
                         "coverage_95": round(cov, 3), "mean_width": round(width, 4)})
            r = rows[-1]
            print(f"  {label:<24} lam {lam:.1f}  est {r['mean_estimate']:+.4f}  "
                  f"bias {r['bias']:+.4f}  coverage {cov:.1%}  width {width:.3f}  "
                  f"({len(res)} reps)")
    d = pd.DataFrame(rows)
    d.to_csv(os.path.join(OUT, "spatial_bootstrap_coverage.csv"), index=False)
    worst = d["coverage_95"].min()
    print(f"\nworst coverage across designs: {worst:.1%} against a nominal 95%")
    print("Wrote spatial_bootstrap_coverage.csv")


if __name__ == "__main__":
    main()
