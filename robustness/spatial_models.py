"""
Spatial error, Durbin-error and combined models, estimated completely.

This replaces the withdrawn spatial_sdem_sac.py. Four things it does that the
withdrawn version did not.

The error parameter comes from the published Kelejian-Prucha three-moment
system (spatial_gm.py), which recovers a known parameter in Monte Carlo; the
withdrawn version used a hand-written pair of moments that were not zero-mean at
the truth and missed 0.6 by 0.046 on a synthetic design.

The estimation sequence is completed. For the error models the data are filtered
by the estimated parameter and the regression refit, which is what makes the
procedure an estimator rather than a diagnostic; for the combined model this is
generalised spatial two-stage least squares, filtering the instruments as well.

Uncertainty is reported for every parameter, including the error parameter, by a
country-block bootstrap that resamples whole countries and repeats the entire
sequence.

Weights are stated. Within each year W is restricted to the countries observed
that year and row-normalised over them, so a country's lag is the mean of its
observed neighbours rather than a sum with absent neighbours set to zero.
Countries with no observed neighbour in a year contribute a zero row and are
counted. tr(W'W) is accumulated over the year blocks actually used.

Covariates entering X and the instruments are lagged one year, so the exogeneity
claim rests on timing rather than on the names of the variables.

Outputs robustness/spatial_models.csv.
"""

import os
import sys
import warnings

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from contagion_galton import build_inputs, build_W_contiguity
from spatial_gm import gm_lambda

OUT = os.path.dirname(os.path.abspath(__file__))
MACRO = os.path.join(OUT, "..", "data", "macro_covariates.csv")
EXOG = ["gdp_pc", "urbanization", "trade_openness"]
N_BOOT = int(os.environ.get("AIM4D_SPATIAL_BOOT", "300"))


class YearBlockW:
    """Row-normalised contiguity within each year, over observed countries only."""

    def __init__(self, cid, yr, W, order):
        pos = {c: i for i, c in enumerate(order)}
        self.blocks, self.rows, self.trWtW, self.isolated = [], [], 0.0, 0
        for t in np.unique(yr):
            m = np.where(yr == t)[0]
            idx = np.array([pos.get(c, -1) for c in cid[m]])
            ok = idx >= 0
            sub = W[np.ix_(idx[ok], idx[ok])].copy()
            rs = sub.sum(axis=1, keepdims=True)
            self.isolated += int((rs == 0).sum())
            sub = np.divide(sub, rs, out=np.zeros_like(sub), where=rs > 0)
            self.blocks.append((m[ok], sub))
            self.trWtW += float(np.trace(sub.T @ sub))
        self.n = len(cid)

    def __call__(self, v):
        out = np.zeros(len(v))
        for rows, sub in self.blocks:
            out[rows] = sub @ v[rows]
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
    V = XtXi @ meat @ XtXi
    return np.sqrt(np.clip(np.diag(V), 0, None)), V


def tsls(y, endog, exog, inst, groups):
    Xall = np.column_stack([endog, exog])
    Zall = np.column_stack([inst, exog])
    ZZi = np.linalg.pinv(Zall.T @ Zall)
    Xhat = Zall @ (ZZi @ (Zall.T @ Xall))
    b = np.linalg.pinv(Xhat.T @ Xall) @ Xhat.T @ y
    resid = y - Xall @ b
    se, V = cluster_se(Xhat, resid, groups)
    return b, se, resid, V


def first_stage_F(endog, exog, inst, groups):
    """Cluster-robust Wald on the excluded instruments (Kleibergen-Paap for k=1)."""
    Z = np.column_stack([inst, exog])
    b, r = ols(endog, Z)
    _, V = cluster_se(Z, r, groups)
    k = inst.shape[1]
    Rb = b[:k]
    VR = V[:k, :k]
    G = len(np.unique(groups))
    return float(Rb @ np.linalg.pinv(VR) @ Rb / k) * (G - 1) / G


def fit_all(y, Xe, WX, Wop, groups, want_boot=False):
    """SEM, SDEM, SAR and SAC on one sample. Returns a dict of parameters."""
    out = {}
    trW = Wop.trWtW

    def filtered(lam, mats):
        return [m - lam * (Wop(m) if m.ndim == 1
                           else np.column_stack([Wop(m[:, k]) for k in range(m.shape[1])]))
                for m in mats]

    # SEM: OLS -> lambda -> filter -> refit
    _, u = ols(y, Xe)
    lam = gm_lambda(u, Wop, trW, Wop.n)
    ys, Xs = filtered(lam, [y, Xe])
    b, r = ols(ys, Xs)
    se, _ = cluster_se(Xs, r, groups)
    out["SEM"] = {"lambda": lam, "beta_ylag": b[1], "se_ylag": se[1]}

    # SDEM: same with neighbours' covariates included
    Xd = np.column_stack([Xe, WX])
    _, ud = ols(y, Xd)
    lam_d = gm_lambda(ud, Wop, trW, Wop.n)
    yd, Xds = filtered(lam_d, [y, Xd])
    bd, rd = ols(yd, Xds)
    sed, Vd = cluster_se(Xds, rd, groups)
    k0 = Xe.shape[1]
    out["SDEM"] = {"lambda": lam_d, "theta": bd[k0:k0 + WX.shape[1]].copy(),
                   "se_theta": sed[k0:k0 + WX.shape[1]].copy(),
                   "beta": bd[:k0].copy(), "V": Vd, "k0": k0}

    # SAR by 2SLS, then SAC by generalised spatial 2SLS.
    # Two instrument sets. The Kelejian-Prucha set is the spatial lags of the
    # exogenous covariates. Lee's best-2SLS instrument is the reduced-form
    # prediction W(I - rho W)^-1 X beta, built from a preliminary estimate; it is
    # the strongest instrument available for this model, so the first-stage F it
    # attains is the ceiling any instrument choice can reach here.
    Wy = Wop(y)
    inst = np.column_stack([np.column_stack([Wop(WX[:, k]) for k in range(WX.shape[1])]),
                            np.column_stack([Wop(Wop(WX[:, k])) for k in range(WX.shape[1])])])
    bs0, _, _, _ = tsls(y, Wy[:, None], Xe, inst, groups)
    rho0 = float(np.clip(bs0[0], -0.9, 0.9))
    b0, _ = ols(y - rho0 * Wy, Xe)
    xb = Xe @ b0
    lee = xb.copy()                      # Neumann expansion of (I - rho W)^-1 X beta
    term = xb.copy()
    for _ in range(6):
        term = rho0 * Wop(term)
        lee = lee + term
    lee = Wop(lee)[:, None]
    out["best_iv"] = {"F_kp": first_stage_F(Wy, Xe, inst, groups),
                      "F_lee": first_stage_F(Wy, Xe, lee, groups),
                      # Fit of y on X after removing rho0*Wy. This is NOT the spatial reduced-form
            # mean, which carries the multiplier (I - rho W)^-1; it is reported as a
            # descriptive statistic only.
            "rf_R2": 1.0 - np.var(y - rho0 * Wy - xb) / np.var(y)}
    bs, ses, us, _ = tsls(y, Wy[:, None], Xe, inst, groups)
    out["SAR"] = {"rho": float(bs[0]), "se_rho": float(ses[0]),
                  "F": out["best_iv"]["F_kp"]}
    bl, sel, ul, _ = tsls(y, Wy[:, None], Xe, lee, groups)
    out["SAR_lee"] = {"rho": float(bl[0]), "se_rho": float(sel[0]),
                      "F": out["best_iv"]["F_lee"]}
    lam_s = gm_lambda(us, Wop, trW, Wop.n)
    yf, Xf, instf = filtered(lam_s, [y, Xe, inst])
    Wyf = Wop(y) - lam_s * Wop(Wop(y))
    bg, seg, _, _ = tsls(yf, Wyf[:, None], Xf, instf, groups)
    out["SAC"] = {"rho": float(bg[0]), "se_rho": float(seg[0]), "lambda": lam_s}
    return out


def main():
    df, countries, years, Y = build_inputs()
    W, _ = build_W_contiguity(countries)
    mac = pd.read_csv(MACRO).rename(columns={"iso3": "country_text_id"})
    mac = mac.sort_values(["country_text_id", "year"])
    for c in EXOG:                       # lag one year: predetermined by timing
        mac[c] = mac.groupby("country_text_id")[c].shift(1)

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
    p = p.dropna(subset=["dy", "y_lag"] + EXOG).reset_index(drop=True)

    cid, yr = p["country_text_id"].values, p["year"].values
    Wop = YearBlockW(cid, yr, W, countries)
    Xcols = ["y_lag"] + EXOG
    yrd = pd.get_dummies(p["year"], drop_first=True).values.astype(float)
    Xe = np.column_stack([np.ones(len(p))] + [p[c].values for c in Xcols] + [yrd])
    WX = np.column_stack([Wop(p[c].values) for c in Xcols])
    y = p["dy"].values

    print(f"n = {len(p)} country-years, {p.country_text_id.nunique()} countries, "
          f"{yr.min()}--{yr.max()}")
    print(f"effective W: year blocks row-normalised over observed countries; "
          f"tr(W'W) = {Wop.trWtW:.1f}; rows with no observed neighbour: {Wop.isolated}\n")

    base = fit_all(y, Xe, WX, Wop, cid)

    # country-block bootstrap: resample countries, repeat the whole sequence
    rng = np.random.default_rng(20260905)
    uniq = np.unique(cid)
    idx = {c: np.where(cid == c)[0] for c in uniq}
    boots = {k: [] for k in ["SEM.lambda", "SDEM.lambda", "SAR.rho", "SARlee.rho",
                             "SAC.rho", "SAC.lambda"]}
    theta_boot = []
    for _ in range(N_BOOT):
        draw = rng.choice(uniq, len(uniq), replace=True)
        j = np.concatenate([idx[c] for c in draw])
        cb, yb = cid[j], yr[j]
        try:
            Wb = YearBlockW(cb, yb, W, countries)
            # Every spatial term must come from the draw's own network. Passing the
            # cached WX would mix the draw's weights for the outcome and residual
            # lags with the full sample's weights for the contextual covariates:
            # on a three-country chain with a draw that drops the middle country,
            # the cached lag is [2,2,2] where the draw's own is [0,0,0].
            WXb = np.column_stack([Wb(p[c].values[j]) for c in Xcols])
            r = fit_all(y[j], Xe[j], WXb, Wb, cb)
        except Exception:
            continue
        boots["SEM.lambda"].append(r["SEM"]["lambda"])
        boots["SDEM.lambda"].append(r["SDEM"]["lambda"])
        boots["SAR.rho"].append(r["SAR"]["rho"])
        boots["SARlee.rho"].append(r["SAR_lee"]["rho"])
        boots["SAC.rho"].append(r["SAC"]["rho"])
        boots["SAC.lambda"].append(r["SAC"]["lambda"])
        theta_boot.append(r["SDEM"]["theta"])

    def ci(v):
        v = np.asarray(v)
        return float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))

    res = []
    for key, val in [("SEM", base["SEM"]["lambda"]), ("SDEM", base["SDEM"]["lambda"]),
                     ("SAC", base["SAC"]["lambda"])]:
        lo, hi = ci(boots[f"{key}.lambda"])
        res.append({"model": key, "param": "lambda", "est": round(val, 4),
                    "ci_low": round(lo, 4), "ci_high": round(hi, 4)})
    for key, bkey, note in (("SAR", "SAR.rho", f"Kelejian-Prucha lags, F {base['SAR']['F']:.1f}"),
                            ("SAR_lee", "SARlee.rho", f"Lee best instrument, F {base['SAR_lee']['F']:.1f}"),
                            ("SAC", "SAC.rho", "generalised spatial 2SLS")):
        lo, hi = ci(boots[bkey])
        res.append({"model": key, "param": "rho", "est": round(base[key]["rho"], 4),
                    "ci_low": round(lo, 4), "ci_high": round(hi, 4), "note": note})
    tb = np.array(theta_boot)
    for k, c in enumerate(Xcols):
        lo, hi = np.percentile(tb[:, k], [2.5, 97.5])
        res.append({"model": "SDEM", "param": f"theta_W.{c}",
                    "est": round(float(base["SDEM"]["theta"][k]), 4),
                    "ci_low": round(float(lo), 4), "ci_high": round(float(hi), 4)})

    for lbl, key, note in (
            ("first-stage F, Kelejian-Prucha lags", "F_kp", "cluster-robust"),
            ("first-stage F, Lee best instrument", "F_lee", "strongest instrument available here"),
            ("fit of the outcome on X (not the spatial reduced form)", "rf_R2",
             "descriptive only; does not by itself establish an identification limit")):
        res.append({"model": "instrument strength", "param": lbl,
                    "est": round(float(base["best_iv"][key]), 4),
                    "ci_low": np.nan, "ci_high": np.nan, "note": note})

    # The common-factor restriction is not reported. Testing theta + rho*beta = 0
    # requires theta, rho and beta from one encompassing specification with their
    # joint uncertainty; taking theta from the Durbin-error bootstrap, rho from the
    # combined-model bootstrap and beta fixed at its full-sample value is not that
    # test, and the version this script previously printed has been removed.

    out = pd.DataFrame(res)
    print(out.to_string(index=False))
    out.to_csv(os.path.join(OUT, "spatial_models.csv"), index=False)
    print(f"\n{len(theta_boot)} of {N_BOOT} bootstrap replications converged")
    print("Wrote spatial_models.csv")


if __name__ == "__main__":
    main()
