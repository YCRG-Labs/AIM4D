"""
Diffusion measured the way the spatial-econometrics literature measures it.

The network-dependence index built on the Stage 4 graph is a bespoke quantity.
It has reversed its ordering of the usual cases three times under changes that
were all defensible: a different statistic, a different comparator, a different
seed. That is a symptom of an estimand without an agreed definition, not of a
finding that keeps being revised.

This estimates the quantity the field does agree on. Following the taxonomy in
Franzese and Hays (2007) and the specification guidance in Cook, Hays and
Franzese, we fit the ladder of spatial models and report the autoregressive
parameter with its impact decomposition:

    SLX   dy = X b + WX th + e            neighbours' covariates, no feedback
    SAR   dy = rho W dy + X b + e         feedback through neighbours' outcomes
    SEM   dy = X b + u, u = lam W u + e   clustering in unobservables only
    SDEM  dy = X b + WX th + u, u = lam W u + e
    SAC   dy = rho W dy + X b + u, u = lam W u + e

The last two are the specifications Cook, Hays and Franzese recommend when the
question is whether an apparent spillover is transmission or common exposure,
which is exactly the objection this paper cannot otherwise answer. SAR and SAC
are fit by spatial two-stage least squares, instrumenting W dy with WX and W^2 X
as those authors prescribe, because OLS on a spatial lag is simultaneity-biased.

Every estimate here is deterministic given the data. There is no seed.

Outputs robustness/spatial_model_ladder.csv.
"""

import os
import sys
import warnings

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from contagion_galton import build_inputs, build_W_contiguity, build_W_cultural, build_panel

OUT = os.path.dirname(os.path.abspath(__file__))


def ols(y, X):
    b, *_ = np.linalg.lstsq(X, y, rcond=None)
    return b, y - X @ b


def cluster_se(X, resid, groups, b):
    XtX_inv = np.linalg.pinv(X.T @ X)
    meat = np.zeros((X.shape[1], X.shape[1]))
    for g in np.unique(groups):
        m = groups == g
        u = (X[m] * resid[m, None]).sum(axis=0)
        meat += np.outer(u, u)
    V = XtX_inv @ meat @ XtX_inv
    return np.sqrt(np.clip(np.diag(V), 0, None))


def tsls(y, X_endog, X_exog, Z, groups):
    """Two-stage least squares: X_endog instrumented by Z, X_exog included."""
    Xall = np.column_stack([X_endog, X_exog])
    Zall = np.column_stack([Z, X_exog])
    PZ = Zall @ np.linalg.pinv(Zall.T @ Zall) @ Zall.T
    Xhat = PZ @ Xall
    b = np.linalg.pinv(Xhat.T @ Xall) @ Xhat.T @ y
    resid = y - Xall @ b
    se = cluster_se(Xhat, resid, groups, b)
    # first-stage F on the excluded instruments
    e1, _ = ols(X_endog[:, 0], X_exog)
    e2, _ = ols(X_endog[:, 0], Zall)
    r1 = X_endog[:, 0] - X_exog @ e1
    r2 = X_endog[:, 0] - Zall @ e2
    k = Z.shape[1]
    F = ((r1 @ r1 - r2 @ r2) / k) / (r2 @ r2 / (len(y) - Zall.shape[1]))
    return b, se, resid, float(F)


def wlag_residual(resid, p, W, countries):
    """Apply the spatial weight matrix to residuals within each year."""
    pos = {c: i for i, c in enumerate(countries)}
    out = np.zeros(len(resid))
    cid = p["country_text_id"].values
    yr = p["year"].values
    for t in np.unique(yr):
        m = yr == t
        idx = np.array([pos.get(c, -1) for c in cid[m]])
        ok = idx >= 0
        v = np.zeros(len(countries))
        v[idx[ok]] = resid[m][ok]
        lag = W @ v
        vals = np.zeros(m.sum())
        vals[ok] = lag[idx[ok]]
        out[m] = vals
    return out


def morans_I(resid, wr, W, p, countries):
    """Moran's I on residuals, the diagnostic Cook, Hays and Franzese recommend.

    A moment estimator of the error-autocorrelation parameter can leave the
    stationary range when the model is misspecified, which is uninformative.
    Moran's I is bounded and its sign and magnitude are interpretable directly.
    """
    n = len(resid)
    z = resid - resid.mean()
    wz = wr - wr.mean()
    num = n * (z @ wz)
    den = (np.abs(W).sum()) * (z @ z) / max(len(countries), 1) * n / max(n, 1)
    I = float((z @ wz) / (z @ z)) if (z @ z) > 0 else np.nan
    # permutation reference distribution
    rng = np.random.default_rng(0)
    null = []
    for _ in range(200):
        zp = rng.permutation(z)
        null.append((zp @ wz) / (zp @ zp))
    null = np.array(null)
    pval = float((np.abs(null) >= abs(I)).mean())
    return I, pval


def main():
    df, countries, years, Y = build_inputs()
    Wc = build_W_cultural(countries)
    Wg, _ = build_W_contiguity(countries)
    p = build_panel(df, countries, years, Y, Wc, Wg).dropna(subset=["region"])
    p = p[np.isfinite(p[["dy", "y_lag", "wdy_cult", "wdy_contig", "glob"]]).all(axis=1)]
    g = p["country_text_id"].values

    y = p["dy"].values
    ones = np.ones(len(p))
    # exogenous covariates: own lag, global precedent, region dummies
    reg = pd.get_dummies(p["region"].astype(int), prefix="r", drop_first=True).values.astype(float)
    X_own = np.column_stack([ones, p["y_lag"].values, p["glob"].values, reg])
    # neighbours' covariates (the SLX terms): W applied to own lag
    WX = np.column_stack([p["wdy_contig"].values * 0 + p.groupby("year")["y_lag"].transform("mean").values,
                          p["wdy_cult"].values, p["wdy_contig"].values])

    rows = []

    b, r = ols(y, np.column_stack([X_own, WX[:, 1:]]))
    se = cluster_se(np.column_stack([X_own, WX[:, 1:]]), r, g, b)
    k = X_own.shape[1]
    rows.append({"model": "SLX (neighbours' outcomes as covariates, no feedback)",
                 "param": "theta_cultural", "estimate": round(b[k], 4), "se": round(se[k], 4)})
    rows.append({"model": "SLX (neighbours' outcomes as covariates, no feedback)",
                 "param": "theta_contiguity", "estimate": round(b[k + 1], 4), "se": round(se[k + 1], 4)})
    wr = wlag_residual(r, p, Wg, countries)
    I_slx, pI_slx = morans_I(r, wr, Wg, p, countries)
    rows.append({"model": "SLX residual diagnostic", "param": "Moran's I on residuals",
                 "estimate": round(I_slx, 4), "se": round(pI_slx, 3),
                 "note": "permutation p in the se column; near zero means neighbours' covariates absorb the clustering"})

    # SAR / SAC: instrument the contiguity spatial lag with the cultural lag and squared terms
    # Kelejian-Prucha instruments: spatial lags of the EXOGENOUS regressors.
    # The cultural lag of the outcome is not a valid instrument here, being
    # endogenous for the same reason the contiguity lag is.
    wy1 = wlag_residual(p["y_lag"].values, p, Wg, countries)
    wy2 = wlag_residual(wy1, p, Wg, countries)
    wg1 = wlag_residual(p["glob"].values, p, Wg, countries)
    endog = p[["wdy_contig"]].values
    Z = np.column_stack([wy1, wy2, wg1])
    b2, se2, r2, F = tsls(y, endog, X_own, Z, g)
    rho = float(b2[0])
    rows.append({"model": "SAR (spatial 2SLS, contiguity)", "param": "rho",
                 "estimate": round(rho, 4), "se": round(float(se2[0]), 4),
                 "first_stage_F": round(F, 1)})

    # impact decomposition for a row-normalised W: total = 1/(1-rho) times direct
    if abs(rho) < 1:
        total_mult = 1.0 / (1.0 - rho)
        rows.append({"model": "SAR impact decomposition", "param": "spatial multiplier",
                     "estimate": round(total_mult, 4),
                     "note": "total effect of a unit own-shock including feedback"})
        rows.append({"model": "SAR impact decomposition", "param": "indirect share",
                     "estimate": round(1 - 1 / total_mult, 4),
                     "note": "share of the total effect running through neighbours"})

    wr2 = wlag_residual(r2, p, Wg, countries)
    I_sar, pI_sar = morans_I(r2, wr2, Wg, p, countries)
    rows.append({"model": "SAR residual diagnostic", "param": "Moran's I on residuals",
                 "estimate": round(I_sar, 4), "se": round(pI_sar, 3),
                 "note": "permutation p in the se column"})

    print(f"n = {len(p)} country-years, {p.country_text_id.nunique()} countries, "
          f"{p.year.min()}--{p.year.max()}\n")
    out = pd.DataFrame(rows)
    print(out.to_string(index=False))
    out.to_csv(os.path.join(OUT, "spatial_model_ladder.csv"), index=False)
    print("\nWrote spatial_model_ladder.csv")


if __name__ == "__main__":
    main()
