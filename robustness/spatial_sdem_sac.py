"""
SDEM and SAC estimated properly.

The withdrawn attempt in spatial_model_ladder.py failed three ways: it regressed
the outcome change on neighbours' outcome changes while calling that a lagged-
covariate model, it instrumented the spatial lag with a term built from a
leave-one-out mean of the outcome, and it selected between specifications by
residual Moran's I. This replaces all three.

The lagged covariates are W applied to genuinely exogenous or predetermined
variables only: the lagged level of the democratic factor and the macroeconomic
covariates, never the contemporaneous outcome change. The instruments for the
autoregressive term are the second and third spatial lags of those same
exogenous covariates, following Kelejian and Prucha; nothing built from the
outcome enters the instrument set. Selection between specifications is by a
common-factor test rather than by a residual diagnostic: SDEM nests SEM, so the
lagged-covariate coefficients are tested jointly, and SAC nests SAR and SEM.

    SEM   dy = X b + u,             u = lam W u + e
    SDEM  dy = X b + WX th + u,     u = lam W u + e
    SAR   dy = rho W dy + X b + e
    SAC   dy = rho W dy + X b + u,  u = lam W u + e

lambda is estimated by the Kelejian-Prucha generalised moments estimator, which
is a moment condition on the residuals rather than a likelihood, and the model is
then refit on spatially filtered data.

Outputs robustness/spatial_sdem_sac.csv.
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
