"""
Replace the logit-ratio network index with a total-variation sensitivity measure.

Three defects motivate this. The published index is a ratio of L1 norms taken on
pre-softmax logits, and softmax is invariant to adding a constant to every
logit, so two representations giving identical predicted probabilities can give
indices differing by more than an order of magnitude; the quantity is therefore
not identified. It is unsigned, so it cannot separate network association with
democratization from network association with erosion. And its counterfactual
zeroes the spatial lags and deletes the edge set, an intervention far outside
the support of the data.

The replacement fixes all three. Total variation between the predicted regime
distributions is invariant to logit shifts by construction, is bounded in [0,1]
without any normalisation, and is read directly as how far the predicted
distribution moves. The counterfactual is the model's own ego-only head, a
trained predictor that never receives network inputs, rather than a
network-trained model evaluated off its data manifold. A signed companion
reports the direction of that movement along the regime ordering, so a country
whose network exposure is associated with democratization is distinguishable
from one whose exposure is associated with erosion.

Both distributions are recovered exactly from the stored residuals: the Stage 4
run saved y - p_full and y - p_local, and y is the Stage 3 posterior, so
p = y - resid recovers each to machine precision (row sums verified at 1.0000).
No re-run of the graph model is required.

Outputs robustness/netdep_total_variation.csv.
"""

import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

OUT = os.path.dirname(os.path.abspath(__file__))
K = 5


def recover():
    r = pd.read_csv(os.path.join(OUT, "..", "stage4_nscm", "nscm_residuals.csv"))
    s = pd.read_csv(os.path.join(OUT, "..", "stage3_msvar", "country_year_states.csv"))
    pc = [f"prob_state_{k}" for k in range(K)]
    m = r.merge(s[["country_text_id", "year"] + pc], on=["country_text_id", "year"], how="inner")
    Y = m[pc].to_numpy(float)
    Pf = Y - m[[f"nscm_resid_full_{k}" for k in range(K)]].to_numpy(float)
    Pl = Y - m[[f"nscm_resid_domestic_{k}" for k in range(K)]].to_numpy(float)
    assert np.allclose(Pf.sum(1), 1, atol=1e-4) and np.allclose(Pl.sum(1), 1, atol=1e-4)
    return m, Pf, Pl


def main():
    m, Pf, Pl = recover()
    tv = 0.5 * np.abs(Pf - Pl).sum(axis=1)

    # Signed direction along the regime ordering: state 0 is the most democratic,
    # so a positive score means network exposure shifts mass toward autocracy.
    lvl = np.arange(K, dtype=float)
    signed = (Pf * lvl).sum(axis=1) - (Pl * lvl).sum(axis=1)

    out = m[["country_text_id", "year"]].copy()
    out["netdep_tv"] = tv
    out["netdep_signed"] = signed
    cs = pd.read_csv(os.path.join(OUT, "..", "stage4_nscm", "contagion_scores.csv"))
    out = out.merge(cs[["country_text_id", "year", "country_name", "contagion_score"]],
                    on=["country_text_id", "year"], how="left")

    print(f"n = {len(out)}")
    print(f"total variation   min {tv.min():.4f}  median {np.median(tv):.4f}  max {tv.max():.4f}")
    print(f"signed shift      min {signed.min():+.4f}  median {np.median(signed):+.4f}  "
          f"max {signed.max():+.4f}")
    print(f"share with signed > 0 (toward autocracy): {(signed > 0).mean():.3f}")
    print()
    rho = out[["netdep_tv", "contagion_score"]].corr(method="spearman").iloc[0, 1]
    print(f"Spearman rank correlation, TV index vs published logit index: {rho:.3f}")
    print()

    recent = out[out.year == 2025]
    cases = ["Hungary", "Türkiye", "Turkey", "Poland", "Ukraine", "Denmark", "Brazil", "Serbia"]
    sub = recent[recent.country_name.isin(cases)].sort_values("netdep_tv", ascending=False)
    print("2025, selected cases:")
    print(f"  {'country':<24}{'TV':>8}{'signed':>10}{'published':>11}")
    for _, r in sub.iterrows():
        print(f"  {str(r.country_name):<24}{r.netdep_tv:>8.3f}{r.netdep_signed:>+10.3f}"
              f"{r.contagion_score:>11.3f}")

    out.to_csv(os.path.join(OUT, "netdep_total_variation.csv"), index=False)
    print("\nWrote netdep_total_variation.csv")


if __name__ == "__main__":
    main()
