"""Precision-at-top-k restricted to genuinely-novel onsets.

A watchlist that scores already-autocratizing countries highly earns easy credit:
naming a regime whose episode began years before the forecast origin is not a
prediction. The accountability claim rests on the harder denominator, precision
among countries that were NOT already in an autocratization episode at the
forecast origin (Ward, Greenhill & Bakke 2010; the ViEWS evaluation philosophy,
Hegre et al. 2019). We therefore report precision@k both ways on the strict
out-of-sample window (year > cutoff): the full list, and the novel-only list that
removes already-ongoing episodes from both the ranking and the positive set.

This is the retrospective analog of the pre-registered prospective scoreboard: it
is computable now from the held-out 2020-2025 onsets, whereas the 2026-2031 list
can only be scored after the fact.
"""

import os
import sys

import numpy as np
import pandas as pd

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
from stage5_ews.estimate import KNOWN_EPISODES, TRAIN_CUTOFF

EWS = os.path.join(REPO, "stage5_ews", "ews_signals.csv")
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "novel_precision_results.csv")
KS = [5, 10, 15, 20, 25, 30]


def precision_at_k(ranked, positives, k):
    top = ranked[:k]
    if not top:
        return np.nan
    return sum(1 for c in top if c in positives) / len(top)


def main():
    ews = pd.read_csv(EWS)
    oos_end = int(ews["year"].max())
    oos = ews[ews["year"] > TRAIN_CUTOFF].copy()

    risk = (oos.groupby("country_name")["combined_risk"].max()
            .sort_values(ascending=False))
    ranked_all = list(risk.index)

    already_ongoing = {c for c, e in KNOWN_EPISODES.items() if e["onset"] <= TRAIN_CUTOFF}
    novel_onset = {c for c, e in KNOWN_EPISODES.items()
                   if TRAIN_CUTOFF < e["onset"] <= oos_end}

    any_episode = already_ongoing | novel_onset
    ranked_novel = [c for c in ranked_all if c not in already_ongoing]

    n = len(ranked_all)
    n_novel_list = len(ranked_novel)
    br_full = sum(1 for c in ranked_all if c in any_episode) / n
    br_novel = (sum(1 for c in ranked_novel if c in novel_onset) / n_novel_list
                if n_novel_list else np.nan)

    print("=" * 72)
    print(f"NOVEL-ONLY PRECISION@K  (strict OOS {TRAIN_CUTOFF+1}-{oos_end})")
    print("=" * 72)
    print(f"  countries ranked: {n}  (novel-eligible after removing "
          f"{len(already_ongoing & set(ranked_all))} already-ongoing: {n_novel_list})")
    print(f"  novel onsets in window: {len(novel_onset)}   "
          f"already-ongoing episodes: {len(already_ongoing)}")
    print(f"  base rate (full / novel-only): {br_full:.3f} / {br_novel:.3f}")
    print()
    print(f"  {'k':>4}{'prec@k full':>16}{'prec@k novel':>16}{'lift novel':>14}")
    rows = []
    for k in KS:
        pf = precision_at_k(ranked_all, any_episode, k)
        pn = precision_at_k(ranked_novel, novel_onset, k)
        lift = pn / br_novel if (br_novel and not np.isnan(br_novel)) else np.nan
        print(f"  {k:>4}{pf:>16.3f}{pn:>16.3f}{lift:>14.1f}")
        rows.append({"k": k, "prec_full": pf, "prec_novel": pn,
                     "base_rate_full": br_full, "base_rate_novel": br_novel,
                     "lift_novel": lift})

    print("\n  Novel onsets and whether the model ranked them in its novel top-10:")
    novel_top10 = set(ranked_novel[:10])
    for c in sorted(novel_onset):
        rank = ranked_novel.index(c) + 1 if c in ranked_novel else None
        flag = "HIT@10" if c in novel_top10 else f"rank {rank}" if rank else "not ranked"
        print(f"    {c:<24} onset {KNOWN_EPISODES[c]['onset']}   [{flag}]")

    pd.DataFrame(rows).to_csv(OUT, index=False)
    print(f"\nSaved to {OUT}")


if __name__ == "__main__":
    main()
