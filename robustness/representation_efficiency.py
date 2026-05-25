"""Does the factor representation earn its keep? The VALID test: data efficiency.

Any test that predicts a V-Dem-derived target favors the raw 332 indicators by
construction (the factor bottleneck is lossy). The structure's genuine advantage
is regularization in the small-sample regime (Stock-Watson 2002; Bernanke-Boivin-
Eliasz 2005 FAVAR): a low-dimensional representation has lower variance, so it
should degrade less than the raw indicators as training positives become scarce,
even though raw wins asymptotically.

Design: same learner (HistGBM), 4 factors vs 332 raw indicators, strict 2019
hold-out fixed, training episodes subsampled to k in a pre-specified grid with R
random draws each. Factors are extracted on <=2019 only (no hold-out peeking),
then projected onto all rows. Report OOS AUC-PR vs k for both representations;
a crossover (factors > raw at small k) is the clean, non-confounded win.
"""

import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "BLIS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, "4")

import sys
import warnings

warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import average_precision_score

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from stage5_ews.estimate import KNOWN_EPISODES, lead_for
from external_benchmarks import load_indicator_panel
from sanity_checks import fast_loadings

OUTPUT_DIR = os.path.dirname(os.path.abspath(__file__))
RNG = np.random.default_rng(42)
K_GRID = [6, 12, 18, 24, None]
R = 12
CUTOFF = 2019

PREONSET, POSTONSET = {}, set()
for c, info in KNOWN_EPISODES.items():
    for y in range(info["onset"] - lead_for(info), info["onset"] + 1):
        PREONSET[(c, y)] = info["onset"]
    for y in range(info["onset"] + 1, info["onset"] + 6):
        POSTONSET.add((c, y))


def label_for(country, year, episodes):
    o = PREONSET.get((country, year))
    return 1 if (o is not None and any(KNOWN_EPISODES[e]["onset"] == o and e == country for e in episodes)) else 0


def main():
    print("=" * 70)
    print("REPRESENTATION VALUE (valid test): data efficiency, factors vs raw 332")
    print("=" * 70)
    panel, indicators = load_indicator_panel()
    panel = panel.reset_index(drop=True)
    yrs = panel["year"].values
    cn = panel["country_name"].values

    scaler = StandardScaler().fit(panel.loc[yrs <= CUTOFF, indicators].values)
    Xstd = scaler.transform(panel[indicators].values)
    L = fast_loadings(Xstd[yrs <= CUTOFF], 4)[0]
    F = Xstd @ L @ np.linalg.inv(L.T @ L)

    post = np.array([(c, y) in POSTONSET for c, y in zip(cn, yrs)])
    train_mask = yrs <= CUTOFF
    test_mask = (yrs > CUTOFF) & (~post)
    y_test = np.array([1 if (c, y) in PREONSET else 0 for c, y in zip(cn[test_mask], yrs[test_mask])])

    train_eps = [c for c, info in KNOWN_EPISODES.items() if info["onset"] <= CUTOFF]
    print(f"  {len(train_eps)} training episodes (onset <= {CUTOFF}); "
          f"test positives = {int(y_test.sum())} post-{CUTOFF} country-years\n")

    def gbm():
        return HistGradientBoostingClassifier(max_iter=200, max_depth=3, learning_rate=0.05,
                                              random_state=0)

    rows = []
    for k in K_GRID:
        kk = len(train_eps) if k is None else k
        ap_f, ap_r = [], []
        for _ in range(R):
            eps = list(RNG.choice(train_eps, size=kk, replace=False))
            ytr = np.array([label_for(c, y, eps) for c, y in zip(cn[train_mask], yrs[train_mask])])
            if ytr.sum() < 2:
                continue
            for X, store in [(F, ap_f), (Xstd, ap_r)]:
                m = gbm().fit(X[train_mask], ytr)
                p = m.predict_proba(X[test_mask])[:, 1]
                store.append(average_precision_score(y_test, p))
        rows.append({"k_episodes": kk, "factors_aucpr": np.mean(ap_f), "factors_sd": np.std(ap_f),
                     "raw_aucpr": np.mean(ap_r), "raw_sd": np.std(ap_r),
                     "delta": np.mean(ap_f) - np.mean(ap_r)})
        print(f"  k={kk:2d} episodes:  4-factors AUC-PR={np.mean(ap_f):.3f}  "
              f"raw-332 AUC-PR={np.mean(ap_r):.3f}  delta={np.mean(ap_f)-np.mean(ap_r):+.3f}")

    df = pd.DataFrame(rows)
    cross = df[df["delta"] > 0]["k_episodes"].max() if (df["delta"] > 0).any() else None
    print(f"\n  factors beat raw at k <= {cross} episodes" if cross else
          "\n  factors do not beat raw at any tested k")
    df.to_csv(os.path.join(OUTPUT_DIR, "representation_efficiency_results.csv"), index=False)
    print(f"Saved to robustness/representation_efficiency_results.csv")


if __name__ == "__main__":
    main()
