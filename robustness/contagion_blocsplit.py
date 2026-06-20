"""Correct diffusion test for the cultural channel: near-vs-far pool decomposition.

The Galton horse race (contagion_galton.py) pits a cultural spatial lag against a
GLOBAL precedent term, but the global term is the mean change over ALL other
countries, which CONTAINS the culturally-near ones. The two regressors overlap and
the broader global term wins by collinearity, so that test cannot isolate whether
diffusion is stronger along cultural lines.

The standard fix decomposes the world into DISJOINT pools: the culturally-near set
(cultural-bloc ties) and the culturally-far set (everyone else). A common global
shock hits both pools equally and nets out in the contrast. The cultural channel is
real, as a RELATIVE claim, iff near transmits significantly more than far
(beta_near > beta_far). We test contemporaneous comovement and a one-year-lagged
diffusion spec (neighbours at t-1 predict own change at t), post-2005, clustered by
country, with a Wald test of beta_near = beta_far.
"""

import os
import sys
import warnings

warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from causal_real_data import FACT, STATE

OUTPUT_DIR = os.path.dirname(os.path.abspath(__file__))
CULTURAL = os.path.join(REPO, "data", "cultural_pairs.csv")
VDEM = os.path.join(REPO, "data", "vdem_v16.csv")
SPLIT_YEAR = 2005


def build_inputs():
    fac = pd.read_csv(FACT)[["country_text_id", "year", "factor_1"]]
    stt = pd.read_csv(STATE)[["country_text_id", "year", "state"]]
    df = fac.merge(stt, on=["country_text_id", "year"], how="inner").dropna()
    countries = sorted(df["country_text_id"].unique())
    years = sorted(df["year"].unique())
    Y = df.pivot(index="country_text_id", columns="year", values="factor_1").reindex(countries).values
    return df, countries, years, Y


def build_pools(countries):
    cp = pd.read_csv(CULTURAL)
    idx = {c: i for i, c in enumerate(countries)}
    n = len(countries)
    A = np.zeros((n, n))
    for a, b in cp[["iso3_a", "iso3_b"]].itertuples(index=False):
        if a in idx and b in idx:
            A[idx[a], idx[b]] = 1
            A[idx[b], idx[a]] = 1
    near = A.copy()
    far = 1.0 - A
    np.fill_diagonal(far, 0.0)

    def rn(W):
        rs = W.sum(axis=1, keepdims=True)
        rs[rs == 0] = 1
        return W / rs
    return rn(near), rn(far)


def build_panel(df, countries, years, Y, Wn, Wf, lag=0):
    rows = []
    for ti in range(1, len(years)):
        t = years[ti]
        dy = Y[:, ti] - Y[:, ti - 1]
        si = ti - 1 - lag
        if si < 1:
            src = np.zeros(len(countries))
        else:
            sd = Y[:, si] - Y[:, si - 1]
            src = np.where(np.isnan(sd), 0.0, sd)
        wn, wf = Wn @ src, Wf @ src
        for i, c in enumerate(countries):
            if np.isnan(dy[i]) or np.isnan(Y[i, ti - 1]):
                continue
            rows.append((c, t, dy[i], Y[i, ti - 1], wn[i], wf[i]))
    p = pd.DataFrame(rows, columns=["country_text_id", "year", "dy", "y_lag", "near", "far"])
    p = p.merge(df[["country_text_id", "year", "state"]], on=["country_text_id", "year"])
    reg = pd.read_csv(VDEM, low_memory=False, usecols=["country_text_id", "year", "e_regionpol_6C"])
    reg = reg.rename(columns={"e_regionpol_6C": "region"}).dropna()
    reg["region"] = reg["region"].astype(int)
    return p.merge(reg, on=["country_text_id", "year"], how="left")


def run(df, countries, years, Y, Wn, Wf, lag, label):
    p = build_panel(df, countries, years, Y, Wn, Wf, lag=lag)
    post = p[(p["year"] >= SPLIT_YEAR) & p["region"].notna()].copy()
    post["region"] = post["region"].astype(int)
    m = smf.ols("dy ~ y_lag + near + far + C(state) + C(region)", data=post).fit(
        cov_type="cluster", cov_kwds={"groups": post["country_text_id"]})
    bn, bf = m.params["near"], m.params["far"]
    pn, pf = m.pvalues["near"], m.pvalues["far"]
    wald = m.t_test("near - far = 0")
    diff_p = float(np.ravel(wald.pvalue))
    print(f"  {label}: near={bn:+.4f}(p={pn:.3f})  far={bf:+.4f}(p={pf:.3f})  "
          f"near-far={bn - bf:+.4f}(p={diff_p:.3f})")
    return {f"{label}_near": bn, f"{label}_near_p": pn, f"{label}_far": bf,
            f"{label}_far_p": pf, f"{label}_diff_p": diff_p}


def main():
    print("=" * 70)
    print("CULTURAL DIFFUSION: NEAR vs FAR POOL DECOMPOSITION (post-2005)")
    print("=" * 70)
    df, countries, years, Y = build_inputs()
    Wn, Wf = build_pools(countries)
    print(f"  {len(countries)} countries; near = cultural-bloc pool, far = all others\n")
    out = {}
    out.update(run(df, countries, years, Y, Wn, Wf, 0, "contemporaneous"))
    out.update(run(df, countries, years, Y, Wn, Wf, 1, "lag1_diffusion "))
    contemp_ok = out["contemporaneous_diff_p"] < 0.05 and out["contemporaneous_near"] > out["contemporaneous_far"]
    lag_ok = out["lag1_diffusion _diff_p"] < 0.05 and out["lag1_diffusion _near"] > out["lag1_diffusion _far"]
    print()
    if contemp_ok or lag_ok:
        print("  => SURVIVES: culturally-near states transmit significantly more than culturally-far.")
        print("     Defensible as a relative claim (diffusion is stronger along cultural lines).")
    else:
        print("  => Does NOT survive: near and far transmit alike. Diffusion is global, not cultural. Demote.")
    pd.DataFrame([out]).to_csv(os.path.join(OUTPUT_DIR, "contagion_blocsplit_results.csv"), index=False)
    print(f"\nSaved to robustness/contagion_blocsplit_results.csv")


if __name__ == "__main__":
    main()
