"""
Diffusion: what the spatial-econometrics ladder would require, and what this
script does not yet deliver.

WITHDRAWN. An earlier version of this script reported a comparison between an
"SLX" and a "SAR" specification and drew a substantive conclusion from their
residual diagnostics. That conclusion is withdrawn. Three defects made it
invalid, and they are recorded here rather than removed so the error is legible:

  1. The model labelled SLX regressed the outcome change on neighbours' outcome
     changes, W*dy, not on spatially lagged explanatory variables, W*X. Those are
     spatial lags of the outcome and are simultaneously determined, so the
     specification was a spatial-autoregressive model under a different name and
     could not contrast contextual effects with outcome transmission.
  2. The instrument set included the spatial lag of a same-year leave-one-out
     mean of the outcome. A neighbour's leave-one-out mean contains the receiving
     country's own outcome, so the instrument was contaminated by the regressand.
     A large first-stage F says nothing about exogeneity.
  3. Choosing the specification whose residual Moran's I lies closer to zero is
     not a valid selection rule among contextual, autoregressive and error
     specifications, and Cook, Hays and Franzese caution specifically against
     using general residual tests this way. The SDEM and SAC models that would
     discriminate among these were never estimated.

The permutation reference for Moran's I was also wrong, holding the observed
spatial lag fixed while permuting the residuals; it is corrected below and the
corrected diagnostic is reported for description only.

What remains here is a descriptive spatial-lag fit with its simultaneity
acknowledged. It supports no verdict about transmission versus common exposure.
Doing that properly requires W*X constructed from exogenous covariates,
instruments that exclude the outcome, and estimation of SDEM and SAC.
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


def morans_I(resid, p, W, countries, n_perm=999, seed=0):
    """Moran's I on residuals with a correctly permuted reference distribution.

    Under the permutation null the spatial lag must be recomputed from the
    permuted values. Holding the observed lag fixed while shuffling the residuals
    produces a null that is too narrow and p-values that are anti-conservative.
    """
    z = resid - resid.mean()
    wz = wlag_residual(z, p, W, countries)
    denom = z @ z
    if denom <= 0:
        return np.nan, np.nan
    I = float((z @ wz) / denom)
    rng = np.random.default_rng(seed)
    null = np.empty(n_perm)
    for b in range(n_perm):
        zp = rng.permutation(z)
        wzp = wlag_residual(zp, p, W, countries)
        null[b] = (zp @ wzp) / (zp @ zp)
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
    I_slx, pI_slx = morans_I(r, p, Wg, countries)
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

    I_sar, pI_sar = morans_I(r2, p, Wg, countries)
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
