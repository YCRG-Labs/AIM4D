"""
Build the Goldstone et al. (2010) PITF five-category regime-type variable from
raw Polity5 components (EXREC, PARCOMP), using the EXREC x PARCOMP cross-tab
from Ulfelder's Early Warning Project replication code.

Categories (reference = full autocracy):
  A/F  full autocracy
  A/P  partial autocracy
  D/fact  partial democracy with factionalism   (the standout PITF predictor)
  D/P  partial democracy without factionalism
  D/F  full democracy
  transition  Polity special codes (-88/-77/-66) and unmatched

Polity5 ends in 2018; the panel runs to 2025. Regime type is the slowest-moving
PITF input, so we carry the last observed category forward per country (LOCF)
through the panel end and flag the imputed tail.

Output: data/pitf_regime.csv  (COWcode, year, pol_cat_pitf, regime_imputed)
"""

import os
import numpy as np
import pandas as pd

DATA = os.path.dirname(os.path.abspath(__file__))


def classify(exrec, parcomp):
    """Ulfelder EWP EXREC x PARCOMP cross-tab. Applied in sequence; later
    assignments overwrite earlier ones, matching the original R code."""
    cat = None
    if exrec in (1, 2, 3, 4, 5, 6) and parcomp in (1, 2):
        cat = "A/F"
    if exrec in (1, 2, 3, 4, 5, 6) and parcomp in (0, 3, 4, 5):
        cat = "A/P"
    if exrec in (7, 8) and parcomp in (1, 2):
        cat = "A/P"
    if parcomp == 3 and exrec in (7, 8):
        cat = "D/fact"
    if exrec == 8 and parcomp in (0, 4):
        cat = "D/P"
    if exrec == 7 and parcomp in (0, 4, 5):
        cat = "D/P"
    if exrec == 8 and parcomp == 5:
        cat = "D/F"
    return cat if cat is not None else "transition"


def main():
    pol = pd.read_csv(os.path.join(DATA, "polity5.csv"))
    pol = pol[pol["year"] >= 1965].copy()
    pol["pol_cat_pitf"] = [classify(e, p) for e, p in zip(pol["exrec"], pol["parcomp"])]
    pol = pol.rename(columns={"ccode": "COWcode"})[["COWcode", "year", "pol_cat_pitf"]]

    panel_years = range(int(pol["year"].min()), 2026)
    out = []
    for cow, g in pol.groupby("COWcode"):
        g = g.set_index("year").reindex(panel_years)
        observed = g["pol_cat_pitf"].notna()
        g["regime_imputed"] = (~observed).astype(int)
        g["pol_cat_pitf"] = g["pol_cat_pitf"].ffill()
        g["COWcode"] = cow
        out.append(g.reset_index().rename(columns={"index": "year"}))
    res = pd.concat(out, ignore_index=True)
    res = res.dropna(subset=["pol_cat_pitf"])
    res = res[(res["year"] >= 1970) & (res["year"] <= 2025)]
    res["COWcode"] = res["COWcode"].astype(int)

    res.to_csv(os.path.join(DATA, "pitf_regime.csv"), index=False)
    print(f"wrote data/pitf_regime.csv: {len(res)} country-years, "
          f"{res['year'].min()}-{res['year'].max()}")
    print("\ncategory distribution:")
    print(res["pol_cat_pitf"].value_counts())
    print(f"\nimputed (post-2018 LOCF tail): {res['regime_imputed'].sum()} rows "
          f"({100*res['regime_imputed'].mean():.1f}%)")


if __name__ == "__main__":
    main()
