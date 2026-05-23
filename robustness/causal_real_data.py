"""Real-data causal evidence for the structural contagion coefficient alpha.

Three pieces backing Proposition 1 on the actual V-Dem panel (not simulation):

  1. Structural estimate of alpha. Outcome = Stage-1 democratic factor_1.
     Spec: y_{i,t} = c + phi*y_{i,t-1} + alpha*(W y)_{i,t-1} + yearFE + stateFE
     with the spatial term LAGGED (predetermined, dissolves Manski reflection),
     own-lag for persistence, year FE to absorb the global autocratization-wave
     common shock (the correlated-effects confounder), regime-state FE, and
     country-clustered standard errors.

  2. Cinelli-Hazlett (2020) robustness value for the untestable neighborhood-
     unconfoundedness assumption, computed in closed form from the t-statistic:
     the minimum partial-R^2 an unobserved confounder must share with BOTH the
     spatial term and the outcome to drive alpha to zero (RV_q1) or to
     insignificance (RV at the 5% level).

  3. Node-permutation placebo: reassign W's topology to random countries and
     re-estimate alpha. If geography carries real contagion, the true-W estimate
     sits in the tail of the permuted null. Rules out the common-shock objection.

Output: robustness/causal_real_data.csv + stdout.
"""
import os
import sys
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FACT = os.path.join(REPO, "stage1_factors", "country_year_factors.csv")
STATE = os.path.join(REPO, "stage3_msvar", "country_year_states.csv")
MAP = os.path.join(REPO, "data", "cow_iso3_mapping.csv")
CONTIG = os.path.join(REPO, "data", "contiguity", "DirectContiguity320", "contdird.csv")
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "causal_real_data.csv")
N_PERM = 500
RNG = np.random.default_rng(42)


def build_W(countries):
    cow = pd.read_csv(MAP).set_index("COWcode")["country_text_id"].to_dict()
    cont = pd.read_csv(CONTIG)
    cont = cont[cont["conttype"] <= 2]
    idx = {c: i for i, c in enumerate(countries)}
    W = np.zeros((len(countries), len(countries)))
    for a, b in cont[["state1no", "state2no"]].drop_duplicates().itertuples(index=False):
        ca, cb = cow.get(a), cow.get(b)
        if ca in idx and cb in idx:
            W[idx[ca], idx[cb]] = 1
            W[idx[cb], idx[ca]] = 1
    rs = W.sum(axis=1, keepdims=True)
    rs[rs == 0] = 1
    return W / rs


def rv_cinelli_hazlett(t, dof, t_crit=1.96):
    f = abs(t) / np.sqrt(dof)
    rv_q1 = 0.5 * (np.sqrt(f**4 + 4 * f**2) - f**2)
    fa = (abs(t) - t_crit) / np.sqrt(dof)
    rv_a = 0.5 * (np.sqrt(fa**4 + 4 * fa**2) - fa**2) if fa > 0 else 0.0
    partial_r2 = t**2 / (t**2 + dof)
    return rv_q1, rv_a, partial_r2


def year_demean(v, years):
    out = v.astype(float).copy()
    for y in np.unique(years):
        m = years == y
        out[m] = out[m] - out[m].mean()
    return out


def main():
    fac = pd.read_csv(FACT)[["country_text_id", "year", "factor_1"]]
    st = pd.read_csv(STATE)[["country_text_id", "year", "state"]]
    df = fac.merge(st, on=["country_text_id", "year"], how="inner").dropna()

    countries = sorted(df["country_text_id"].unique())
    W = build_W(countries)
    cidx = {c: i for i, c in enumerate(countries)}
    n_edges = int((W > 0).sum())
    print("=" * 70)
    print("Real-data causal evidence for structural contagion coefficient alpha")
    print("=" * 70)
    print(f"  Panel: {len(countries)} countries, years {df.year.min()}-{df.year.max()}, "
          f"{len(df)} country-years")
    print(f"  W: contiguity (conttype<=2), {n_edges} directed edges, row-normalized")

    years = sorted(df["year"].unique())
    Y = df.pivot(index="country_text_id", columns="year", values="factor_1").reindex(countries)
    Ymat = Y.values

    rows = []
    for ti in range(1, len(years)):
        t, tprev = years[ti], years[ti - 1]
        yprev = Ymat[:, ti - 1]
        wy_prev = W @ np.where(np.isnan(yprev), 0.0, yprev)
        for i, c in enumerate(countries):
            yt = Ymat[i, ti]
            if np.isnan(yt) or np.isnan(yprev[i]):
                continue
            rows.append((c, t, yt, yprev[i], wy_prev[i]))
    panel = pd.DataFrame(rows, columns=["country_text_id", "year", "y", "own_lag", "wy_lag"])
    panel = panel.merge(df[["country_text_id", "year", "state"]], on=["country_text_id", "year"])

    print(f"  Regression sample: {len(panel)} country-years\n")

    print("-" * 70)
    print("[1] Structural estimate (year FE + state FE, country-clustered SE)")
    print("-" * 70)
    m = smf.ols("y ~ own_lag + wy_lag + C(year) + C(state)", data=panel).fit(
        cov_type="cluster", cov_kwds={"groups": panel["country_text_id"]})
    alpha = m.params["wy_lag"]
    se = m.bse["wy_lag"]
    t = m.tvalues["wy_lag"]
    p = m.pvalues["wy_lag"]
    dof = int(m.df_resid)
    print(f"    alpha_hat (contagion) = {alpha:+.4f}")
    print(f"    cluster-robust SE     = {se:.4f}")
    print(f"    t                     = {t:.3f}   p = {p:.2e}")
    print(f"    own-lag (persistence) = {m.params['own_lag']:+.4f}")
    print(f"    n = {int(m.nobs)}, dof_resid = {dof}, clusters = {panel.country_text_id.nunique()}")

    print("\n" + "-" * 70)
    print("[2] Cinelli-Hazlett (2020) sensitivity to unobserved confounding")
    print("-" * 70)
    rv1, rva, pr2 = rv_cinelli_hazlett(t, dof)
    print(f"    partial R^2 of contagion term         = {pr2:.4f}")
    print(f"    Robustness Value RV_(q=1)             = {rv1:.4f}")
    print(f"      => a confounder must explain >= {100*rv1:.1f}% of the residual")
    print(f"         variation in BOTH the spatial term and the outcome to drive")
    print(f"         alpha to zero.")
    print(f"    Robustness Value RV_(q=1, alpha=.05)  = {rva:.4f}")
    print(f"      => and >= {100*rva:.1f}% to render it statistically insignificant.")

    print("\n" + "-" * 70)
    print(f"[3] Node-permutation placebo ({N_PERM} permutations of W's topology)")
    print("-" * 70)
    yv = panel["y"].values
    ov = panel["own_lag"].values
    yr = panel["year"].values
    ci = panel["country_text_id"].map(cidx).values
    yi = np.array([years.index(y) for y in panel["year"].values])
    yd = year_demean(yv, yr)
    od = year_demean(ov, yr)
    yd_r = yd - od * (np.dot(od, yd) / np.dot(od, od))

    def alpha_for_W(Wm):
        wyp = np.zeros(len(panel))
        lag_cache = {}
        for ti in range(1, len(years)):
            lag_cache[ti] = Wm @ np.where(np.isnan(Ymat[:, ti - 1]), 0.0, Ymat[:, ti - 1])
        for k in range(len(panel)):
            wyp[k] = lag_cache[yi[k]][ci[k]]
        wd = year_demean(wyp, yr)
        wd_r = wd - od * (np.dot(od, wd) / np.dot(od, od))
        return np.dot(wd_r, yd_r) / np.dot(wd_r, wd_r)

    alpha_fwl = alpha_for_W(W)
    null = np.empty(N_PERM)
    for k in range(N_PERM):
        perm = RNG.permutation(len(countries))
        null[k] = alpha_for_W(W[perm][:, perm])
    pval = (np.sum(np.abs(null) >= abs(alpha_fwl)) + 1) / (N_PERM + 1)
    print(f"    true-W alpha (FWL)        = {alpha_fwl:+.4f}")
    print(f"    permuted-W null: mean={null.mean():+.4f} sd={null.std():.4f} "
          f"95th-pct|.|={np.percentile(np.abs(null),95):.4f}")
    print(f"    placebo p-value           = {pval:.4f}  "
          f"({'PASS: real W in tail' if pval < 0.05 else 'NOT in tail'})")

    pd.DataFrame([{
        "alpha_hat": alpha, "se": se, "t": t, "p": p, "dof": dof,
        "own_lag": m.params["own_lag"], "n": int(m.nobs),
        "partial_r2": pr2, "rv_q1": rv1, "rv_q1_a05": rva,
        "placebo_alpha_fwl": alpha_fwl, "placebo_null_mean": null.mean(),
        "placebo_null_sd": null.std(), "placebo_pvalue": pval,
    }]).to_csv(OUT, index=False)
    print(f"\nWrote {OUT}")


if __name__ == "__main__":
    main()
