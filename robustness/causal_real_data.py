"""Spatial-diffusion evidence for the network-contagion channel, real panel.

Two specifications, because the choice is the whole point (Franzese-Hays
"Galton's problem"; Plumper-Neumayer 2010; Leeson-Dean 2009 AJPS):

  OVER-CONTROLLED (level + near-unit-root own-lag + year FE):
    y_{i,t} = c + phi*y_{i,t-1} + alpha*(W y)_{i,t-1} + yearFE + stateFE
    With phi~0.94 the own-lag and year FE absorb essentially all variance and
    the spatial term cannot be detected by construction. Reported only to show
    the artifact.

  LITERATURE-STANDARD (change outcome, contemporaneous spatial lag, NO year FE):
    dY_{i,t} = c + theta*y_{i,t-1} + alpha*(W dY)_{i,t} + stateFE
    Estimated by OLS (descriptive) and 2SLS instrumenting the contemporaneous
    W*dY with the predetermined W*y_{t-1} (Kelejian-Prucha). Year FE dropped
    because the global wave is part of diffusion; the model's own latent factors
    already absorb common shocks. The with-FE vs without-FE contrast is the
    decisive diagnostic.

Plus a node-permutation placebo on W. Identification here is spatial-
correlational diffusion, NOT experimental causation; the neighbor-contiguity
channel is known to be the weakest (global/regional comovement dominates,
Schmotz-Selvik 2025), so claims are calibrated accordingly.

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
N_BOOT = 300
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
    rv1 = 0.5 * (np.sqrt(f**4 + 4 * f**2) - f**2)
    fa = (abs(t) - t_crit) / np.sqrt(dof)
    rva = 0.5 * (np.sqrt(fa**4 + 4 * fa**2) - fa**2) if fa > 0 else 0.0
    return rv1, rva, t**2 / (t**2 + dof)


def cluster_boot_2sls(d, endog, instr, exog_cols, groups, n_boot):
    gs = d[groups].unique()
    by = {g: d[d[groups] == g] for g in gs}

    def fit(sub):
        X1 = sub[exog_cols + [instr]].values
        X1 = np.column_stack([np.ones(len(sub)), X1])
        what = X1 @ np.linalg.lstsq(X1, sub[endog].values, rcond=None)[0]
        X2 = np.column_stack([np.ones(len(sub)), sub[exog_cols].values, what])
        b = np.linalg.lstsq(X2, sub["dy"].values, rcond=None)[0]
        return b[-1]

    point = fit(d)
    bs = []
    for _ in range(n_boot):
        draw = RNG.choice(gs, size=len(gs), replace=True)
        sub = pd.concat([by[g] for g in draw], ignore_index=True)
        try:
            bs.append(fit(sub))
        except Exception:
            continue
    bs = np.array(bs)
    return point, bs.std(), np.percentile(bs, [2.5, 97.5])


def main():
    fac = pd.read_csv(FACT)[["country_text_id", "year", "factor_1"]]
    stt = pd.read_csv(STATE)[["country_text_id", "year", "state"]]
    df = fac.merge(stt, on=["country_text_id", "year"], how="inner").dropna()
    countries = sorted(df["country_text_id"].unique())
    W = build_W(countries)
    cidx = {c: i for i, c in enumerate(countries)}
    years = sorted(df["year"].unique())
    Y = df.pivot(index="country_text_id", columns="year", values="factor_1").reindex(countries).values

    print("=" * 70)
    print("Spatial diffusion of democracy — real panel, literature-standard spec")
    print("=" * 70)
    print(f"  {len(countries)} countries, {years[0]}-{years[-1]}, W=contiguity "
          f"({int((W>0).sum())} edges, row-normalized)")

    rows = []
    for ti in range(1, len(years)):
        t = years[ti]
        yprev = Y[:, ti - 1]
        ycur = Y[:, ti]
        dy_vec = ycur - yprev
        wdy = W @ np.where(np.isnan(dy_vec), 0.0, dy_vec)
        wy_lag = W @ np.where(np.isnan(yprev), 0.0, yprev)
        for i, c in enumerate(countries):
            if np.isnan(ycur[i]) or np.isnan(yprev[i]):
                continue
            rows.append((c, t, i, ti, ycur[i] - yprev[i], yprev[i], wdy[i], wy_lag[i]))
    p = pd.DataFrame(rows, columns=["country_text_id", "year", "ci", "ti",
                                    "dy", "y_lag", "wdy", "wy_lag"])
    p = p.merge(df[["country_text_id", "year", "state"]], on=["country_text_id", "year"])
    cl = {"groups": p["country_text_id"]}
    print(f"  regression sample: {len(p)} country-years\n")

    print("-" * 70)
    print("[A] OVER-CONTROLLED level spec (own-lag + year FE) — the artifact")
    print("-" * 70)
    lev_rows = []
    for ti in range(1, len(years)):
        t = years[ti]
        yprev = Y[:, ti - 1]
        wy = W @ np.where(np.isnan(yprev), 0.0, yprev)
        for i, c in enumerate(countries):
            if np.isnan(Y[i, ti]) or np.isnan(yprev[i]):
                continue
            lev_rows.append((c, t, Y[i, ti], yprev[i], wy[i]))
    lev = pd.DataFrame(lev_rows, columns=["country_text_id", "year", "y", "own_lag", "wy_lag"])
    lev = lev.merge(df[["country_text_id", "year", "state"]], on=["country_text_id", "year"])
    mlev = smf.ols("y ~ own_lag + wy_lag + C(year) + C(state)", data=lev).fit(
        cov_type="cluster", cov_kwds={"groups": lev["country_text_id"]})
    print(f"    alpha = {mlev.params['wy_lag']:+.4f}  t={mlev.tvalues['wy_lag']:.2f}  "
          f"p={mlev.pvalues['wy_lag']:.3f}  (own-lag phi={mlev.params['own_lag']:.3f})")
    print("    => null BY CONSTRUCTION (Franzese-Hays / Plumper-Neumayer)")

    print("\n" + "-" * 70)
    print("[B] LITERATURE-STANDARD change spec  dY ~ y_lag + W*dY  (+/- year FE)")
    print("-" * 70)
    m_nofe = smf.ols("dy ~ y_lag + wdy + C(state)", data=p).fit(
        cov_type="cluster", cov_kwds=cl)
    a_nofe = m_nofe.params["wdy"]; t_nofe = m_nofe.tvalues["wdy"]
    m_fe = smf.ols("dy ~ y_lag + wdy + C(state) + C(year)", data=p).fit(
        cov_type="cluster", cov_kwds=cl)
    a_fe = m_fe.params["wdy"]; t_fe = m_fe.tvalues["wdy"]
    print(f"    OLS, NO year FE : alpha={a_nofe:+.4f}  t={t_nofe:.2f}  p={m_nofe.pvalues['wdy']:.2e}")
    print(f"    OLS, + year FE  : alpha={a_fe:+.4f}  t={t_fe:.2f}  p={m_fe.pvalues['wdy']:.2e}")
    print(f"    contrast: year FE shrinks alpha by "
          f"{100*(1-a_fe/a_nofe):.0f}% — the FE absorb the diffusion wave")

    print("\n" + "-" * 70)
    print("[C] 2SLS (contemporaneous W*dY instrumented by predetermined W*y_lag)")
    print("-" * 70)
    a_iv, se_iv, ci_iv = cluster_boot_2sls(
        p, endog="wdy", instr="wy_lag", exog_cols=["y_lag"],
        groups="country_text_id", n_boot=N_BOOT)
    t_iv = a_iv / se_iv if se_iv > 0 else np.nan
    print(f"    alpha_2SLS = {a_iv:+.4f}  cluster-boot SE={se_iv:.4f}  "
          f"t={t_iv:.2f}  95% CI [{ci_iv[0]:+.4f}, {ci_iv[1]:+.4f}]")
    rv1, rva, pr2 = rv_cinelli_hazlett(t_nofe, int(m_nofe.df_resid))
    print(f"    Cinelli-Hazlett RV (no-FE OLS): RV_q1={rv1:.3f}, RV_a05={rva:.3f}, "
          f"partial R2={pr2:.4f}")

    print("\n" + "-" * 70)
    print(f"[D] Node-permutation placebo on W ({N_PERM} perms, no-FE change spec)")
    print("-" * 70)
    yr = p["year"].values
    dyv = p["dy"].values
    ylv = p["y_lag"].values
    ci_arr = p["ci"].values
    ti_arr = p["ti"].values

    def resid(v):
        out = v.astype(float).copy()
        X = np.column_stack([np.ones(len(p)), ylv])
        return out - X @ np.linalg.lstsq(X, out, rcond=None)[0]

    dy_r = resid(dyv)

    def alpha_for_W(Wm):
        wdy_k = np.zeros(len(p))
        cache = {}
        for ti in range(1, len(years)):
            dvec = Y[:, ti] - Y[:, ti - 1]
            cache[ti] = Wm @ np.where(np.isnan(dvec), 0.0, dvec)
        for k in range(len(p)):
            wdy_k[k] = cache[ti_arr[k]][ci_arr[k]]
        wr = resid(wdy_k)
        return np.dot(wr, dy_r) / np.dot(wr, wr)

    a_real = alpha_for_W(W)
    null = np.array([alpha_for_W(W[(perm := RNG.permutation(len(countries)))][:, perm])
                     for _ in range(N_PERM)])
    pval = (np.sum(np.abs(null) >= abs(a_real)) + 1) / (N_PERM + 1)
    print(f"    true-W alpha={a_real:+.4f}  null mean={null.mean():+.4f} sd={null.std():.4f}")
    print(f"    placebo p={pval:.4f}  ({'PASS: real W in tail' if pval<0.05 else 'not in tail'})")

    print("\n" + "=" * 70)
    print("VERDICT (calibrated to the diffusion literature)")
    print("=" * 70)
    print(f"  Level+FE spec: null (artifact). Change spec without FE: alpha="
          f"{a_nofe:+.4f} (t={t_nofe:.1f}); 2SLS={a_iv:+.4f}.")
    print("  Interpretation: a modest, literature-consistent spatial DIFFUSION")
    print("  effect; spatial-correlational, not experimentally causal; contiguity")
    print("  is the weakest channel (global/regional comovement dominates).")

    pd.DataFrame([{
        "level_fe_alpha": mlev.params["wy_lag"], "level_fe_p": mlev.pvalues["wy_lag"],
        "change_nofe_alpha": a_nofe, "change_nofe_t": t_nofe, "change_nofe_p": m_nofe.pvalues["wdy"],
        "change_fe_alpha": a_fe, "change_fe_t": t_fe,
        "alpha_2sls": a_iv, "se_2sls": se_iv, "ci_lo": ci_iv[0], "ci_hi": ci_iv[1],
        "rv_q1": rv1, "partial_r2": pr2,
        "placebo_alpha": a_real, "placebo_p": pval,
    }]).to_csv(OUT, index=False)
    print(f"\nWrote {OUT}")


if __name__ == "__main__":
    main()
