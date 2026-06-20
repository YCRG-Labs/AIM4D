"""Galton's-problem defense for the cultural-bloc diffusion finding.

The objection (and Schmotz-Selvik 2025's central method): a cultural-bloc spatial
lag may pick up shared heritage rather than transmission, and regional/neighbour
effects often vanish once GLOBAL precedent is controlled. We therefore run a
horse race. We re-estimate the recent-era (post-2005) diffusion model adding, as
mutual controls, (i) a geographic/border spatial lag (COW direct land contiguity),
(ii) a leave-one-out global-precedent term (the mean change among all other
countries that year, Schmotz-Selvik's dominant channel), and (iii) region fixed
effects (a proxy for shared structural/colonial/linguistic heritage). The cultural
diffusion association is credible only if its coefficient survives this competition.
If it dies once global precedent enters, the honest conclusion is that the signal
is global, not cultural, and the finding should be demoted.
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
CONTIG = os.path.join(REPO, "data", "contiguity", "DirectContiguity320", "contdir.csv")
COWMAP = os.path.join(REPO, "data", "cow_iso3_mapping.csv")
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


def _rownorm(W):
    rs = W.sum(axis=1, keepdims=True)
    rs[rs == 0] = 1
    return W / rs


def build_W_cultural(countries):
    cp = pd.read_csv(CULTURAL)
    idx = {c: i for i, c in enumerate(countries)}
    W = np.zeros((len(countries), len(countries)))
    for a, b in cp[["iso3_a", "iso3_b"]].itertuples(index=False):
        if a in idx and b in idx:
            W[idx[a], idx[b]] = 1
            W[idx[b], idx[a]] = 1
    return _rownorm(W)


def build_W_contiguity(countries):
    cm = pd.read_csv(COWMAP)
    cow2iso = dict(zip(cm["COWcode"], cm["country_text_id"]))
    cd = pd.read_csv(CONTIG)
    cd = cd[cd["conttype"] == 1]
    idx = {c: i for i, c in enumerate(countries)}
    W = np.zeros((len(countries), len(countries)))
    n = 0
    for lo, hi in cd[["statelno", "statehno"]].itertuples(index=False):
        a, b = cow2iso.get(lo), cow2iso.get(hi)
        if a in idx and b in idx:
            W[idx[a], idx[b]] = 1
            W[idx[b], idx[a]] = 1
            n += 1
    return _rownorm(W), n


def build_panel(df, countries, years, Y, Wc, Wg):
    rows = []
    for ti in range(1, len(years)):
        t = years[ti]
        dy = Y[:, ti] - Y[:, ti - 1]
        dy0 = np.where(np.isnan(dy), 0.0, dy)
        wc, wg = Wc @ dy0, Wg @ dy0
        valid = ~np.isnan(dy)
        tot, cnt = dy0[valid].sum(), valid.sum()
        for i, c in enumerate(countries):
            if np.isnan(dy[i]) or np.isnan(Y[i, ti - 1]):
                continue
            glob = (tot - dy0[i]) / (cnt - 1) if cnt > 1 else 0.0
            rows.append((c, t, dy[i], Y[i, ti - 1], wc[i], wg[i], glob))
    p = pd.DataFrame(rows, columns=["country_text_id", "year", "dy", "y_lag",
                                    "wdy_cult", "wdy_contig", "glob"])
    p = p.merge(df[["country_text_id", "year", "state"]], on=["country_text_id", "year"])
    reg = pd.read_csv(VDEM, low_memory=False, usecols=["country_text_id", "year", "e_regionpol_6C"])
    reg = reg.rename(columns={"e_regionpol_6C": "region"}).dropna()
    reg["region"] = reg["region"].astype(int)
    return p.merge(reg, on=["country_text_id", "year"], how="left")


def fit(formula, data):
    return smf.ols(formula, data=data).fit(cov_type="cluster",
                                           cov_kwds={"groups": data["country_text_id"]})


def main():
    print("=" * 70)
    print("CULTURAL DIFFUSION: GALTON HORSE RACE (post-2005)")
    print("=" * 70)
    df, countries, years, Y = build_inputs()
    Wc = build_W_cultural(countries)
    Wg, n_contig = build_W_contiguity(countries)
    p = build_panel(df, countries, years, Y, Wc, Wg)
    post = p[(p["year"] >= SPLIT_YEAR) & p["region"].notna()].copy()
    post["region"] = post["region"].astype(int)
    print(f"  countries={len(countries)}  contiguity land-border edges={n_contig}  "
          f"post-2005 country-years={len(post)}\n")

    specs = [
        ("M1 cultural only                  ", "dy ~ y_lag + wdy_cult + C(state)"),
        ("M2 + geographic (border) lag       ", "dy ~ y_lag + wdy_cult + wdy_contig + C(state)"),
        ("M3 + global precedent (LOO)        ", "dy ~ y_lag + wdy_cult + wdy_contig + glob + C(state)"),
        ("M4 + region fixed effects          ", "dy ~ y_lag + wdy_cult + wdy_contig + glob + C(state) + C(region)"),
    ]
    out = {}
    for name, f in specs:
        m = fit(f, post)
        a, t, pv = m.params["wdy_cult"], m.tvalues["wdy_cult"], m.pvalues["wdy_cult"]
        extra = ""
        if "wdy_contig" in m.params:
            extra += f"  geo={m.params['wdy_contig']:+.3f}(p={m.pvalues['wdy_contig']:.2f})"
        if "glob" in m.params:
            extra += f"  glob={m.params['glob']:+.3f}(p={m.pvalues['glob']:.2f})"
        flag = "significant" if pv < 0.05 else "NULL"
        print(f"  {name}: cultural alpha={a:+.4f}  t={t:.2f}  p={pv:.3f}  [{flag}]{extra}")
        key = name.split()[0]
        out[f"{key}_cult_alpha"], out[f"{key}_cult_p"] = a, pv
    survives = out["M4_cult_p"] < 0.05
    print(f"\n  => cultural diffusion {'SURVIVES' if survives else 'does NOT survive'} the full horse race "
          f"(geo + global precedent + region FE)")
    print(f"     {'Finding stands as a cultural-channel refinement of Schmotz-Selvik.' if survives else 'Demote: signal is not separably cultural once global precedent is controlled.'}")
    pd.DataFrame([out]).to_csv(os.path.join(OUTPUT_DIR, "contagion_galton_results.csv"), index=False)
    print(f"\nSaved to robustness/contagion_galton_results.csv")


if __name__ == "__main__":
    main()
