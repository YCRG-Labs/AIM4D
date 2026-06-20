"""Within-bloc, size-matched dyadic test for the cultural diffusion channel.

The near-vs-far pool decomposition (contagion_blocsplit.py) is confounded by pool
composition: the "far" pool is every non-bloc country (~135 of them), a large and
heterogeneous set that tracks the global autocratization wave more tightly than a
handful of cultural neighbours simply by aggregation, regardless of any cultural
mechanism. A larger, more diverse pool co-moves with the common wave by
construction, so near < far tells us nothing about culture.

We remove that confound with a matched dyadic design (Neumayer & Plumper 2010 on
directed-dyad spatial lags; Franzese & Hays 2007 on the spatial-lag / Galton
correction; Beck, Gleditsch & Beardsley 2006 on non-geographic connectivity). For
each receiver country we draw a far pool that is (i) the SAME SIZE as its set of
cultural partners and (ii) restricted to the receiver's own political region, so
near and far are drawn from comparable pools. We then regress own change on the
near-partner spatial lag, the matched far-partner lag, and a leave-one-out global
precedent term (the common-wave control is complementary to, not a substitute for,
the matching). The cultural channel is credible as a relative claim only if the
near lag transmits more than the size-matched far lag, averaged over many random
matched draws. Region fixed effects and regime-state controls are retained.
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
N_DRAWS = int(os.environ.get("AIM4D_DYADIC_DRAWS", "200"))
SEED = int(os.environ.get("AIM4D_DYADIC_SEED", "12345"))


def build_inputs():
    fac = pd.read_csv(FACT)[["country_text_id", "year", "factor_1"]]
    stt = pd.read_csv(STATE)[["country_text_id", "year", "state"]]
    df = fac.merge(stt, on=["country_text_id", "year"], how="inner").dropna()
    countries = sorted(df["country_text_id"].unique())
    years = sorted(df["year"].unique())
    Y = df.pivot(index="country_text_id", columns="year", values="factor_1").reindex(countries).values
    reg = pd.read_csv(VDEM, low_memory=False, usecols=["country_text_id", "year", "e_regionpol_6C"])
    reg = reg.rename(columns={"e_regionpol_6C": "region"}).dropna()
    reg["region"] = reg["region"].astype(int)
    region_of = (reg.sort_values("year").groupby("country_text_id")["region"].last().to_dict())
    return df, countries, years, Y, reg, region_of


def cultural_neighbors(countries):
    cp = pd.read_csv(CULTURAL)
    idx = {c: i for i, c in enumerate(countries)}
    nb = {i: set() for i in range(len(countries))}
    for a, b in cp[["iso3_a", "iso3_b"]].itertuples(index=False):
        if a in idx and b in idx:
            nb[idx[a]].add(idx[b])
            nb[idx[b]].add(idx[a])
    return nb


def lag_source(Y, ti, lag):
    si = ti - 1 - lag
    if si < 1:
        return None
    sd = Y[:, si] - Y[:, si - 1]
    return np.where(np.isnan(sd), 0.0, sd)


def build_panel(df, countries, years, Y, nb, region_of, rng, lag):
    n = len(countries)
    region_members = {}
    for i, c in enumerate(countries):
        region_members.setdefault(region_of.get(c), []).append(i)

    rows = []
    for ti in range(1, len(years)):
        t = years[ti]
        dy = Y[:, ti] - Y[:, ti - 1]
        src = lag_source(Y, ti, lag)
        if src is None:
            continue
        valid = ~np.isnan(dy)
        glob_tot, glob_cnt = src[valid].sum(), valid.sum()
        for i, c in enumerate(countries):
            if np.isnan(dy[i]) or np.isnan(Y[i, ti - 1]):
                continue
            near = list(nb[i])
            if not near:
                continue
            k = len(near)
            reg_pool = [j for j in region_members.get(region_of.get(c), [])
                        if j != i and j not in nb[i]]
            pool = reg_pool if len(reg_pool) >= k else [j for j in range(n)
                                                        if j != i and j not in nb[i]]
            if len(pool) < k:
                continue
            far = rng.choice(pool, size=k, replace=False)
            near_lag = float(np.mean(src[near]))
            far_lag = float(np.mean(src[far]))
            glob = (glob_tot - src[i]) / (glob_cnt - 1) if glob_cnt > 1 else 0.0
            rows.append((c, t, dy[i], Y[i, ti - 1], near_lag, far_lag, glob))
    p = pd.DataFrame(rows, columns=["country_text_id", "year", "dy", "y_lag",
                                    "near", "far", "glob"])
    p = p.merge(df[["country_text_id", "year", "state"]], on=["country_text_id", "year"])
    return p


def run_draws(df, countries, years, Y, nb, region_of, reg, lag, label):
    rng = np.random.default_rng(SEED)
    recs = []
    for _ in range(N_DRAWS):
        p = build_panel(df, countries, years, Y, nb, region_of, rng, lag)
        p = p.merge(reg, on=["country_text_id", "year"], how="left")
        post = p[(p["year"] >= SPLIT_YEAR) & p["region"].notna()].copy()
        post["region"] = post["region"].astype(int)
        m = smf.ols("dy ~ y_lag + near + far + glob + C(state) + C(region)", data=post).fit(
            cov_type="cluster", cov_kwds={"groups": post["country_text_id"]})
        diff = m.params["near"] - m.params["far"]
        diff_p = float(np.ravel(m.t_test("near - far = 0").pvalue)[0])
        recs.append((m.params["near"], m.params["far"], diff, diff_p,
                     m.pvalues["near"], m.pvalues["far"]))
    a = np.array(recs)
    near_m, far_m, diff_m = a[:, 0].mean(), a[:, 1].mean(), a[:, 2].mean()
    sig_frac = float(np.mean((a[:, 3] < 0.05) & (a[:, 2] > 0)))
    print(f"  {label}: near={near_m:+.4f}  far(matched)={far_m:+.4f}  "
          f"near-far={diff_m:+.4f}  draws with near>far & p<.05: {sig_frac:.0%}")
    return {f"{label}_near": near_m, f"{label}_far_matched": far_m,
            f"{label}_near_minus_far": diff_m, f"{label}_sig_frac": sig_frac}


def main():
    print("=" * 72)
    print(f"CULTURAL DIFFUSION: WITHIN-REGION SIZE-MATCHED DYADIC TEST "
          f"({N_DRAWS} draws, post-{SPLIT_YEAR})")
    print("=" * 72)
    df, countries, years, Y, reg, region_of = build_inputs()
    nb = cultural_neighbors(countries)
    n_tied = sum(1 for i in nb if nb[i])
    print(f"  {len(countries)} countries; {n_tied} with cultural ties; "
          f"far pool = size-matched, within-region, non-tied\n")
    out = {}
    out.update(run_draws(df, countries, years, Y, nb, region_of, reg, 0, "contemporaneous"))
    out.update(run_draws(df, countries, years, Y, nb, region_of, reg, 1, "lag1_diffusion"))
    print()
    contemp_ok = out["contemporaneous_sig_frac"] >= 0.5
    lag_ok = out["lag1_diffusion_sig_frac"] >= 0.5
    if contemp_ok or lag_ok:
        print("  => SURVIVES the matched design: cultural-near transmits more than a "
              "size-matched within-region far pool.")
    else:
        print("  => Does NOT survive once the far pool is size- and region-matched: the "
              "apparent near<far gap was pool composition, not a cultural channel. "
              "The global-wave horse race (contagion_galton.py) remains the load-bearing test.")
    pd.DataFrame([out]).to_csv(os.path.join(OUTPUT_DIR, "contagion_dyadic_results.csv"), index=False)
    print(f"\nSaved to robustness/contagion_dyadic_results.csv")


if __name__ == "__main__":
    main()
