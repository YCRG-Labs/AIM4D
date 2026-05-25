"""Pre-specified fix attempt for the recent-era contagion signal: re-run the
structural spatial-diffusion model on a CULTURAL-bloc weight matrix instead of
border contiguity. The paper's own GNN gives the four edge types near-equal
weight and the contiguity channel is the weakest; Schmotz-Selvik (2025) argue
backsliding diffuses through broad/cultural comovement, not borders. So the
recent-era null on contiguity may simply be the wrong network.

Cultural-bloc W from cultural_pairs.csv, change-spec dY ~ y_lag + W*dY + C(state),
full sample and split at 2005, plus a node-permutation placebo on the cultural
topology. Cultural blocs share much unobserved history, so this is reported as a
predictive-correlational diffusion association, not a causal effect.
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
SPLIT_YEAR = 2005
N_PERM = 500
RNG = np.random.default_rng(42)


def build_W_cultural(countries):
    cp = pd.read_csv(CULTURAL)
    idx = {c: i for i, c in enumerate(countries)}
    W = np.zeros((len(countries), len(countries)))
    for a, b in cp[["iso3_a", "iso3_b"]].itertuples(index=False):
        if a in idx and b in idx:
            W[idx[a], idx[b]] = 1
            W[idx[b], idx[a]] = 1
    rs = W.sum(axis=1, keepdims=True)
    rs[rs == 0] = 1
    return W / rs


def build_inputs():
    fac = pd.read_csv(FACT)[["country_text_id", "year", "factor_1"]]
    stt = pd.read_csv(STATE)[["country_text_id", "year", "state"]]
    df = fac.merge(stt, on=["country_text_id", "year"], how="inner").dropna()
    countries = sorted(df["country_text_id"].unique())
    years = sorted(df["year"].unique())
    Y = df.pivot(index="country_text_id", columns="year", values="factor_1").reindex(countries).values
    return df, countries, years, Y


def build_panel(df, countries, years, Y, W):
    rows = []
    for ti in range(1, len(years)):
        t = years[ti]
        yprev, ycur = Y[:, ti - 1], Y[:, ti]
        wdy = W @ np.where(np.isnan(ycur - yprev), 0.0, ycur - yprev)
        for i, c in enumerate(countries):
            if np.isnan(ycur[i]) or np.isnan(yprev[i]):
                continue
            rows.append((c, t, i, ti, ycur[i] - yprev[i], yprev[i], wdy[i]))
    p = pd.DataFrame(rows, columns=["country_text_id", "year", "ci", "ti", "dy", "y_lag", "wdy"])
    return p.merge(df[["country_text_id", "year", "state"]], on=["country_text_id", "year"])


def clustered(formula, data):
    return smf.ols(formula, data=data).fit(cov_type="cluster",
                                           cov_kwds={"groups": data["country_text_id"]})


def residualize(v, ylag):
    X = np.column_stack([np.ones(len(v)), ylag])
    return v - X @ np.linalg.lstsq(X, v, rcond=None)[0]


def alpha_for_W(W, p, Y, years):
    cache = {ti: W @ np.where(np.isnan(Y[:, ti] - Y[:, ti - 1]), 0.0, Y[:, ti] - Y[:, ti - 1])
             for ti in range(1, len(years))}
    wdy = np.array([cache[ti][ci] for ci, ti in zip(p["ci"].values, p["ti"].values)])
    dy_r = residualize(p["dy"].values.astype(float), p["y_lag"].values)
    wr = residualize(wdy, p["y_lag"].values)
    return float(np.dot(wr, dy_r) / np.dot(wr, wr))


def main():
    print("=" * 70)
    print("CONTAGION FIX: cultural-bloc W (vs contiguity), + node-permutation placebo")
    print("=" * 70)
    df, countries, years, Y = build_inputs()
    W = build_W_cultural(countries)
    p = build_panel(df, countries, years, Y, W)
    print(f"  cultural W: {int((W > 0).sum())} directed edges over {len(countries)} countries, "
          f"{len(p)} country-years\n")
    out = {}
    for label, sub in [("full   ", p),
                       (f"pre  (<{SPLIT_YEAR})", p[p["year"] < SPLIT_YEAR]),
                       (f"post (>={SPLIT_YEAR})", p[p["year"] >= SPLIT_YEAR])]:
        if sub["state"].nunique() < 2 or len(sub) < 50:
            continue
        m = clustered("dy ~ y_lag + wdy + C(state)", sub)
        a, t, pv = m.params["wdy"], m.tvalues["wdy"], m.pvalues["wdy"]
        print(f"  {label}: alpha={a:+.4f}  t={t:.2f}  p={pv:.3f}  {'(significant)' if pv < 0.05 else ''}")
        key = label.strip().split()[0]
        out[f"cultural_{key}_alpha"] = a
        out[f"cultural_{key}_p"] = pv

    a_real = alpha_for_W(W, p, Y, years)
    null = np.array([alpha_for_W(W[(perm := RNG.permutation(len(countries)))][:, perm], p, Y, years)
                     for _ in range(N_PERM)])
    pval = (np.sum(np.abs(null) >= abs(a_real)) + 1) / (N_PERM + 1)
    print(f"\n  node-permutation placebo ({N_PERM} perms): true alpha={a_real:+.4f}, "
          f"null mean={null.mean():+.4f} sd={null.std():.4f}, p={pval:.4f} "
          f"({'PASS: tied to cultural topology' if pval < 0.05 else 'degree-structure artifact'})")
    out["placebo_alpha"] = a_real
    out["placebo_p"] = pval

    print("\n  [contiguity baseline: pre 0.33 (p<0.001), post 0.04 (p=0.10, null)]")
    print(f"  => {'PASS: cultural channel diffuses in recent era' if out.get('cultural_post_p', 1) < 0.05 else 'no recent-era cultural diffusion'}")
    pd.DataFrame([out]).to_csv(os.path.join(OUTPUT_DIR, "contagion_fix_results.csv"), index=False)
    print(f"\nSaved to robustness/contagion_fix_results.csv")


if __name__ == "__main__":
    main()
