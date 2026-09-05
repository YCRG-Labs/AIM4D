"""
WITHDRAWN. The error-parameter estimator in this script is wrong.

The lagged covariates and instruments here are sound: W is applied only to
predetermined and exogenous variables, and the autoregressive term is
instrumented by second and third spatial lags of those same covariates, so
nothing built from the outcome enters. That part is kept for reuse.

The error parameter is not. kp_lambda minimises the squares of

    g1 = (e'e - (We)'(We)) / n,   g2 = e'We / n,    e = u - lam*W*u

and g1 is not a zero-mean moment at the true parameter. For innovations of
variance s2 its expectation is s2*(1 - tr(W'W)/n), and row standardisation does
not make that trace ratio one. A generalised-moments estimator of a spatial
error process needs the trace correction and the innovation variance; this omits
both. On a row-standardised 100-node cycle with a true parameter of 0.6 and unit
innovations, the population version of this objective is minimised at 0.646.

Two further gaps. The docstring promised a refit on spatially filtered data and
the code never filters or refits, so the reported quantities are ordinary and
two-stage fits with an error parameter computed alongside them, not estimates of
the error model. And the joint F on the lagged covariates tests the nesting of
the lagged-covariate model inside the error model; it is not the common-factor
restriction, which relates the autoregressive, own-covariate and lagged-covariate
coefficients.

No value produced by this script is reported in the paper. Replace kp_lambda
with a validated implementation, complete the filtered refit, and check the
result against a known spatial process before using anything here.
"""

import os
import sys
import warnings

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from contagion_galton import build_inputs, build_W_contiguity

OUT = os.path.dirname(os.path.abspath(__file__))
MACRO = os.path.join(OUT, "..", "data", "macro_covariates.csv")
EXOG = ["gdp_pc", "urbanization", "trade_openness"]


def wlag(vec, cid, yr, W, order):
    pos = {c: i for i, c in enumerate(order)}
    out = np.zeros(len(vec))
    for t in np.unique(yr):
        m = yr == t
        idx = np.array([pos.get(c, -1) for c in cid[m]])
        ok = idx >= 0
        v = np.zeros(len(order))
        v[idx[ok]] = vec[m][ok]
        lag = W @ v
        vals = np.zeros(m.sum())
        vals[ok] = lag[idx[ok]]
        out[m] = vals
    return out


def ols(y, X):
    b = np.linalg.pinv(X.T @ X) @ X.T @ y
    return b, y - X @ b


def cluster_se(X, resid, groups):
    XtXi = np.linalg.pinv(X.T @ X)
    meat = np.zeros((X.shape[1], X.shape[1]))
    for g in np.unique(groups):
        m = groups == g
        u = (X[m] * resid[m, None]).sum(axis=0)
        meat += np.outer(u, u)
    return np.sqrt(np.clip(np.diag(XtXi @ meat @ XtXi), 0, None))


def kp_lambda(u, wu, wwu):
    """Kelejian-Prucha GM estimator: moment conditions on the residual process."""
    n = len(u)
    def obj(lam):
        e = u - lam * wu
        we = wu - lam * wwu
        g1 = e @ e / n - (we @ we) / n
        g2 = e @ we / n
        return g1 ** 2 + g2 ** 2
    r = minimize_scalar(obj, bounds=(-0.95, 0.95), method="bounded")
    return float(r.x)


def tsls(y, endog, exog, inst, groups):
    Xall = np.column_stack([endog, exog])
    Zall = np.column_stack([inst, exog])
    P = Zall @ np.linalg.pinv(Zall.T @ Zall) @ Zall.T
    Xhat = P @ Xall
    b = np.linalg.pinv(Xhat.T @ Xall) @ Xhat.T @ y
    resid = y - Xall @ b
    se = cluster_se(Xhat, resid, groups)
    b1, _ = ols(endog[:, 0], exog)
    b2, _ = ols(endog[:, 0], Zall)
    r1 = endog[:, 0] - exog @ b1
    r2 = endog[:, 0] - Zall @ b2
    k = inst.shape[1]
    F = ((r1 @ r1 - r2 @ r2) / k) / (r2 @ r2 / (len(y) - Zall.shape[1]))
    return b, se, resid, float(F)


def main():
    df, countries, years, Y = build_inputs()
    W, _ = build_W_contiguity(countries)

    mac = pd.read_csv(MACRO).rename(columns={"iso3": "country_text_id"})
    rows = []
    for ti in range(1, len(years)):
        t = years[ti]
        dy = Y[:, ti] - Y[:, ti - 1]
        for i, c in enumerate(countries):
            if np.isnan(dy[i]) or np.isnan(Y[i, ti - 1]):
                continue
            rows.append((c, t, dy[i], Y[i, ti - 1]))
    p = pd.DataFrame(rows, columns=["country_text_id", "year", "dy", "y_lag"])
    p = p.merge(mac[["country_text_id", "year"] + EXOG], on=["country_text_id", "year"], how="left")
    p[EXOG] = p[EXOG].apply(lambda s: (s - s.mean()) / s.std())
    p = p.dropna(subset=["dy", "y_lag"] + EXOG)
    cid, yr = p["country_text_id"].values, p["year"].values

    # exogenous / predetermined covariates only; the outcome never enters X or Z
    Xcols = ["y_lag"] + EXOG
    Xe = np.column_stack([np.ones(len(p))] + [p[c].values for c in Xcols])
    yrd = pd.get_dummies(p["year"], drop_first=True).values.astype(float)
    Xe = np.column_stack([Xe, yrd])                      # year effects absorb the global wave
    WX = np.column_stack([wlag(p[c].values, cid, yr, W, countries) for c in Xcols])
    W2X = np.column_stack([wlag(WX[:, k], cid, yr, W, countries) for k in range(WX.shape[1])])
    W3X = np.column_stack([wlag(W2X[:, k], cid, yr, W, countries) for k in range(W2X.shape[1])])
    y = p["dy"].values
    Wy = wlag(y, cid, yr, W, countries)

    res = []
    # SEM
    b, u = ols(y, Xe)
    lam = kp_lambda(u, wlag(u, cid, yr, W, countries),
                    wlag(wlag(u, cid, yr, W, countries), cid, yr, W, countries))
    res.append({"model": "SEM", "param": "lambda", "est": round(lam, 4), "se": np.nan,
                "note": "clustering in unobservables only"})
    # SDEM
    Xd = np.column_stack([Xe, WX])
    bd, ud = ols(y, Xd)
    sed = cluster_se(Xd, ud, cid)
    lam_d = kp_lambda(ud, wlag(ud, cid, yr, W, countries),
                      wlag(wlag(ud, cid, yr, W, countries), cid, yr, W, countries))
    k0 = Xe.shape[1]
    for j, c in enumerate(Xcols):
        res.append({"model": "SDEM", "param": f"theta_W.{c}", "est": round(bd[k0 + j], 4),
                    "se": round(sed[k0 + j], 4), "note": "neighbours' exogenous covariate"})
    res.append({"model": "SDEM", "param": "lambda", "est": round(lam_d, 4), "se": np.nan,
                "note": "error clustering left after lagged covariates"})
    # SAR and SAC, instruments from exogenous covariates only
    bs, ses, us, F = tsls(y, Wy[:, None], Xe, np.column_stack([W2X, W3X]), cid)
    res.append({"model": "SAR", "param": "rho", "est": round(float(bs[0]), 4),
                "se": round(float(ses[0]), 4), "note": f"spatial 2SLS, first-stage F {F:.1f}"})
    lam_s = kp_lambda(us, wlag(us, cid, yr, W, countries),
                      wlag(wlag(us, cid, yr, W, countries), cid, yr, W, countries))
    res.append({"model": "SAC", "param": "lambda", "est": round(lam_s, 4), "se": np.nan,
                "note": "error clustering left after the autoregressive term"})

    # common-factor test: are the lagged covariates jointly zero (SDEM -> SEM)?
    rss_r = u @ u
    rss_u = ud @ ud
    q = WX.shape[1]
    Fcf = ((rss_r - rss_u) / q) / (rss_u / (len(y) - Xd.shape[1]))
    res.append({"model": "SDEM vs SEM", "param": "joint F on lagged covariates",
                "est": round(float(Fcf), 2), "se": np.nan,
                "note": "large means neighbours' covariates matter beyond error clustering"})

    print(f"n = {len(p)} country-years, {p.country_text_id.nunique()} countries, "
          f"{p.year.min()}--{p.year.max()}, year effects included\n")
    out = pd.DataFrame(res)
    print(out.to_string(index=False))
    out.to_csv(os.path.join(OUT, "spatial_sdem_sac.csv"), index=False)
    print("\nWrote spatial_sdem_sac.csv")


if __name__ == "__main__":
    main()
