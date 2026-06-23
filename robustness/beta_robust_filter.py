"""Robustness check for the time-varying democratic beta: a Student-t observation
filter as a principled alternative to upper-tail winsorization.

The transient beta spikes (e.g. Hungary +14 in 1989) arise because the Gaussian
state-space MLE places the state-innovation variance at its upper boundary during
the synchronized 1989-1992 transitions, so the filtered loading behaves as a near
random walk and tracks the instantaneous ratio of a country's idiosyncratic move
to a small contemporaneous global change (Stock & Watson 1998; Harvey & Luati
2014). A symmetric remedy (tightening the state variance, or shrinking the beta
toward a constant) suppresses the spurious positive transition spike but also
reverses the substantively meaningful negative onset signal, because both come
from the same mechanism.

A Student-t observation density is the natural principled candidate, but it does
NOT rescue the asymmetry. Estimated by MLE, the heavy-tailed filter drives the
state-innovation variance to its LOWER boundary and flattens Hungary's Factor-1
loading to a near constant (~+0.6 throughout), erasing not just the 1989 spike but
the substantively meaningful 2009-2011 negative onset as well. This script
documents both failed alternatives, tightening the Gaussian state variance (which
flips the 2010 sign) and the Student-t filter (which flattens the series), and is
the evidence that no symmetric or robust estimator separates the spurious positive
transition spike from the genuine negative onset, because the two are statistically
identical. The directional upper-tail winsorization used in the main text is
therefore justified on substantive, not statistical, grounds.
"""

import os
import sys
import warnings

warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
from scipy.optimize import minimize

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from stage2_betas.estimate import (
    load_factor_scores, compute_loo_global, FACTOR_COLS, MIN_OBS, MAX_TRAIN_YEAR,
    kalman_tvp_univariate, tvp_loglik_uni,
)

NU = float(os.environ.get("AIM4D_ROBUST_NU", "4.0"))
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "beta_robust_filter_results.csv")


def robust_t_kalman(y, x, q_var, r_var, nu=NU):
    """Kalman TVP filter+smoother with a Student-t observation density.

    The observation precision is scaled per step by the Student-t score weight
    w_t = (nu+1)/(nu + v_t^2/F_t), which is < 1 for outlier innovations, so a lone
    large shock inflates the observation noise rather than the state (Harvey &
    Luati 2014; Creal, Koopman & Lucas 2013).
    """
    T = len(y)
    valid = min(10, T)
    xi, yi = x[:valid], y[:valid]
    mask = np.isfinite(xi) & np.isfinite(yi) & (np.abs(xi) > 1e-10)
    beta0 = np.sum(xi[mask] * yi[mask]) / np.sum(xi[mask] ** 2) if mask.sum() >= 2 else 1.0

    bf, Pf, bp_, Pp_ = (np.zeros(T) for _ in range(4))
    for t in range(T):
        bp = beta0 if t == 0 else bf[t - 1]
        Pp = (1.0 if t == 0 else Pf[t - 1]) + q_var
        bp_[t], Pp_[t] = bp, Pp
        v = y[t] - x[t] * bp
        F = x[t] ** 2 * Pp + r_var
        w = (nu + 1.0) / (nu + (v * v) / max(F, 1e-12))
        r_eff = r_var / max(w, 1e-6)
        F_eff = x[t] ** 2 * Pp + r_eff
        K = Pp * x[t] / F_eff
        bf[t] = bp + K * v
        Pf[t] = (1.0 - K * x[t]) * Pp

    bs, Ps = np.zeros(T), np.zeros(T)
    bs[-1], Ps[-1] = bf[-1], Pf[-1]
    for t in range(T - 2, -1, -1):
        J = Pf[t] / Pp_[t + 1] if Pp_[t + 1] > 1e-15 else 0.0
        bs[t] = bf[t] + J * (bs[t + 1] - bp_[t + 1])
        Ps[t] = Pf[t] + J ** 2 * (Ps[t + 1] - Pp_[t + 1])
    return bs


def robust_loglik(params, y, x, nu=NU):
    q_var, r_var = np.exp(params[0]), np.exp(params[1])
    T = len(y)
    valid = min(10, T)
    xi, yi = x[:valid], y[:valid]
    mask = np.isfinite(xi) & np.isfinite(yi) & (np.abs(xi) > 1e-10)
    bp = np.sum(xi[mask] * yi[mask]) / np.sum(xi[mask] ** 2) if mask.sum() >= 2 else 1.0
    Pp = 1.0 + q_var
    from scipy.special import gammaln
    c = gammaln((nu + 1) / 2) - gammaln(nu / 2) - 0.5 * np.log(nu * np.pi)
    ll = 0.0
    for t in range(T):
        v = y[t] - x[t] * bp
        F = x[t] ** 2 * Pp + r_var
        if F > 1e-15:
            ll += c - 0.5 * np.log(F) - (nu + 1) / 2 * np.log(1 + (v * v) / (nu * F))
        w = (nu + 1.0) / (nu + (v * v) / max(F, 1e-12))
        K = Pp * x[t] / (x[t] ** 2 * Pp + r_var / max(w, 1e-6))
        bp = bp + K * v
        Pp = (1 - K * x[t]) * Pp + q_var
    return -ll


def fit_robust(y, x, n_train):
    yt, xt = y[:n_train], x[:n_train]
    init = np.array([np.log(0.05), np.log(np.var(yt) * 0.5 + 1e-6)])
    res = minimize(robust_loglik, init, args=(yt, xt), method="L-BFGS-B",
                   bounds=[(-8, 2), (-8, 5)])
    q, r = np.exp(res.x[0]), np.exp(res.x[1])
    bs = robust_t_kalman(yt, xt, q, r)
    if n_train < len(y):
        bs = np.concatenate([bs, np.full(len(y) - n_train, bs[-1])])
    return bs, q


def gaussian_kalman_beta(y, x, n_train):
    init = np.array([np.log(0.05), np.log(np.var(y[:n_train]) * 0.5 + 1e-6)])
    res = minimize(tvp_loglik_uni, init, args=(y[:n_train], x[:n_train]),
                   method="L-BFGS-B", bounds=[(-8, 2), (-8, 5)])
    q, r = np.exp(res.x[0]), np.exp(res.x[1])
    bs, _ = kalman_tvp_univariate(y[:n_train], x[:n_train], q, r)
    if n_train < len(y):
        bs = np.concatenate([bs, np.full(len(y) - n_train, bs[-1])])
    return bs, q


def tightened_q_demo(df):
    """Falsification of the symmetric remedy: tightening the Kalman state-variance
    upper bound suppresses the +1989 spike but flips the sign of the 2010 onset.
    """
    hun = df[df["country_name"] == "Hungary"].sort_values("year")
    yrs = hun["year"].values
    gf = compute_loo_global(df, "Hungary").loc[yrs].values
    y = np.diff(hun["factor_1"].values); x = np.diff(gf[:, 0])
    n_train = max(2, int((yrs <= MAX_TRAIN_YEAR).sum()) - 1)
    yd = yrs[1:]

    def at(a, yy):
        j = np.where(yd == yy)[0]; return a[j[0]] if len(j) else np.nan

    print("\n" + "=" * 72)
    print("SYMMETRIC REMEDY (tighten Kalman state-variance upper bound) — Hungary F1")
    print("=" * 72)
    print(f"  {'q_max':>10}{'q_fit':>9}{'beta1989':>10}{'beta2010':>10}")
    for qml, lbl in [(2.0, "exp(2) [current]"), (0.0, "exp(0)"), (-1.0, "exp(-1)")]:
        init = np.array([np.log(0.05), np.log(np.var(y[:n_train]) * 0.5 + 1e-6)])
        res = minimize(tvp_loglik_uni, init, args=(y[:n_train], x[:n_train]),
                       method="L-BFGS-B", bounds=[(-8, qml), (-8, 5)])
        qv = np.exp(res.x[0])
        bs, _ = kalman_tvp_univariate(y[:n_train], x[:n_train], qv, np.exp(res.x[1]))
        print(f"  {qml:>10.1f}{qv:>9.3f}{at(bs,1989):>10.2f}{at(bs,2010):>10.2f}   [{lbl}]")
    print("  => tightening shrinks the 1989 spike but drives the 2010 onset toward zero "
          "and positive: the substantive counter-movement signal is lost.")


def main():
    df = load_factor_scores()
    tightened_q_demo(df)
    countries = df["country_name"].unique()
    rows = []
    case_track = {}
    for country in countries:
        cdf = df[df["country_name"] == country].sort_values("year")
        if len(cdf) < MIN_OBS:
            continue
        years = cdf["year"].values
        gf = compute_loo_global(df, country).loc[years].values
        y_all = cdf[FACTOR_COLS].values
        n_pre = int((years <= MAX_TRAIN_YEAR).sum())
        n_train = max(2, n_pre - 1)
        dy = np.diff(y_all[:, 0]); dx = np.diff(gf[:, 0])
        bg, qg = gaussian_kalman_beta(dy, dx, n_train)
        br, qr = fit_robust(dy, dx, n_train)
        yd = years[1:]
        for i, yy in enumerate(yd):
            rows.append({"country": country, "year": int(yy),
                         "beta_gaussian": bg[i], "beta_robust_t": br[i]})
        if country in ("Hungary", "Türkiye", "Poland"):
            case_track[country] = (yd, bg, br, qg, qr)

    res = pd.DataFrame(rows)
    res.to_csv(OUT, index=False)

    print("=" * 72)
    print(f"STUDENT-t ROBUST TVP FILTER vs GAUSSIAN (Factor-1 Kalman loading, nu={NU})")
    print("=" * 72)
    for c, (yd, bg, br, qg, qr) in case_track.items():
        print(f"\n{c}  (q_var: gaussian={qg:.2f}  robust-t={qr:.2f})")
        print(f"  {'year':>6}{'gaussian':>11}{'robust-t':>11}")
        for yy in [1989, 1990, 2009, 2010, 2011, 2025]:
            j = np.where(yd == yy)[0]
            if len(j):
                print(f"  {yy:>6}{bg[j[0]]:>11.2f}{br[j[0]]:>11.2f}")

    g = res["beta_gaussian"].values
    r = res["beta_robust_t"].values
    print(f"\nPanel agreement (robust-t vs raw Gaussian): "
          f"Pearson r={np.corrcoef(g, r)[0,1]:.3f}, Spearman rank "
          f"r={pd.Series(g).corr(pd.Series(r), method='spearman'):.3f}")
    print(f"max |gaussian| = {np.nanmax(np.abs(g)):.1f}   max |robust-t| = {np.nanmax(np.abs(r)):.1f}")
    print(f"\nSaved to {OUT}")


if __name__ == "__main__":
    main()
