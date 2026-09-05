"""
Why the network ranking reverses, decomposed per country.

The published index is A / (A + B) with A the L1 norm of the logit spillover and
B the L1 norm of the domestic logit vector. Only A is stored, but B is
recoverable exactly as B = A (1 - index) / index, so the index can be split into
the movement it measures and the scale it divides by. That split is the whole
question: a country can score high because its prediction genuinely moves a lot,
or merely because its domestic logits happen to sit near zero.

The scale term is the one with no defensible interpretation. Softmax is
invariant to adding a constant to every logit, so B can be made arbitrarily
large or small without altering a single predicted probability. Any ranking
driven by B is a ranking of an arbitrary representation.

This reports, per country: the movement term, the scale term, their ratio, the
published index, the total-variation index computed on recovered probabilities,
and the signed direction along the regime ordering. It also reports which regime
states the probability mass moves between, which is what "network dependence"
should mean substantively.

Outputs robustness/netdep_reversal_diagnosis.csv.
"""

import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from netdep_total_variation import recover

OUT = os.path.dirname(os.path.abspath(__file__))
K = 5
STATES = ["lib dem", "elec dem", "hybrid", "comp auth", "closed auth"]
CASES = ["Hungary", "Türkiye", "Turkey", "Poland", "Ukraine", "Serbia",
         "Tunisia", "Brazil", "Denmark", "United States of America"]


def main():
    cs = pd.read_csv(os.path.join(OUT, "..", "stage4_nscm", "contagion_scores.csv"))
    sp = cs[[f"spillover_state_{k}" for k in range(K)]].to_numpy(float)
    A = np.abs(sp).sum(axis=1)
    idx = cs["contagion_score"].to_numpy(float)
    with np.errstate(divide="ignore", invalid="ignore"):
        B = np.where(idx > 0, A * (1 - idx) / idx, np.nan)
    cs = cs.assign(move_A=A, scale_B=B, ratio_AB=A / np.where(B > 0, B, np.nan))

    m, Pf, Pl = recover()
    tv = 0.5 * np.abs(Pf - Pl).sum(axis=1)
    lvl = np.arange(K, dtype=float)
    signed = (Pf * lvl).sum(axis=1) - (Pl * lvl).sum(axis=1)
    prob = m[["country_text_id", "year"]].copy()
    prob["tv"] = tv
    prob["signed"] = signed
    for k in range(K):
        prob[f"dP_{k}"] = Pf[:, k] - Pl[:, k]

    d = cs.merge(prob, on=["country_text_id", "year"], how="inner")

    print("=== what drives the PUBLISHED index, across the whole panel ===")
    print(f"  corr(published index, movement A) = "
          f"{d[['contagion_score','move_A']].corr(method='spearman').iloc[0,1]:+.3f}")
    print(f"  corr(published index, scale B)    = "
          f"{d[['contagion_score','scale_B']].corr(method='spearman').iloc[0,1]:+.3f}")
    print(f"  corr(published index, TV)         = "
          f"{d[['contagion_score','tv']].corr(method='spearman').iloc[0,1]:+.3f}")
    print(f"  corr(movement A, TV)              = "
          f"{d[['move_A','tv']].corr(method='spearman').iloc[0,1]:+.3f}")
    print()

    r = d[d.year == 2025]
    sub = r[r.country_name.isin(CASES)].sort_values("contagion_score", ascending=False)
    print("=== 2025, per country ===")
    print(f"  {'country':<22}{'pub idx':>9}{'move A':>9}{'scale B':>10}{'A/B':>8}"
          f"{'TV':>8}{'signed':>9}")
    for _, x in sub.iterrows():
        print(f"  {str(x.country_name)[:21]:<22}{x.contagion_score:>9.3f}{x.move_A:>9.2f}"
              f"{x.scale_B:>10.2f}{x.ratio_AB:>8.2f}{x.tv:>8.3f}{x.signed:>+9.3f}")

    print()
    print("=== where the probability mass actually moves (2025) ===")
    print(f"  {'country':<22}" + "".join(f"{s:>12}" for s in STATES))
    for _, x in sub.iterrows():
        row = "".join(f"{x[f'dP_{k}']:>+12.3f}" for k in range(K))
        print(f"  {str(x.country_name)[:21]:<22}{row}")

    d.to_csv(os.path.join(OUT, "netdep_reversal_diagnosis.csv"), index=False)
    print("\nWrote netdep_reversal_diagnosis.csv")


if __name__ == "__main__":
    main()
