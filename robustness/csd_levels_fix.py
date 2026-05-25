"""Pre-specified fix attempt for the CSD channel: compute critical-slowing-down
indicators on the Gaussian-DETRENDED LEVEL of each factor, not on the differenced
domestic residual. CSD theory (Scheffer 2009, Dakos 2012) concerns rising
variance/autocorrelation in the level as resilience is lost; first-differencing
removes exactly that signal. Detrending (Gaussian kernel, sigma=3) isolates
fluctuations around the slow trend without conditioning on the level itself.

Single pre-specified variant (detrended levels, sigma=3), evaluated on the same
matched detection-vs-false-positive test as csd_hardening. Pass iff the pre-onset
significance rate clearly exceeds the stable false-positive rate.
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
from scipy.ndimage import gaussian_filter1d

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from stage5_ews.estimate import (multivariate_csd, rolling_stats, KNOWN_EPISODES,
                                  lead_for, BASELINE_END, WINDOW, MIN_WINDOW)
from false_positive_analysis import STABLE_DEMOCRACIES
from csd_hardening import trend_sig

OUTPUT_DIR = os.path.dirname(os.path.abspath(__file__))
FACTORS = ["factor_1", "factor_2", "factor_3", "factor_4"]
SIGMA = 3.0
FAC = pd.read_csv(os.path.join(REPO, "stage1_factors", "country_year_factors.csv"))


def detrend(x):
    x = np.asarray(x, float)
    return x - gaussian_filter1d(x, sigma=SIGMA, mode="nearest")


def level_indicators(name, window=WINDOW):
    sub = FAC[FAC["country_name"] == name].sort_values("year")
    if len(sub) < MIN_WINDOW + 3:
        return None
    yrs = sub["year"].values
    M = np.column_stack([detrend(sub[f].values) for f in FACTORS])
    dom_eig, mean_xcorr, total_var = multivariate_csd(M, window=window)
    stat = [rolling_stats(M[:, k], window=window) for k in range(M.shape[1])]
    var = np.nanmean([s[0] for s in stat], axis=0)
    ar1 = np.nanmean([s[1] for s in stat], axis=0)
    return yrs, {"dom_eig": dom_eig, "mean_xcorr": mean_xcorr, "total_var": total_var,
                 "var": var, "ar1": ar1}


def yearrate(pairs, indicator, method="arma", window=WINDOW):
    flagged = total = 0
    for name, lo, hi in pairs:
        ind = level_indicators(name, window=window)
        if ind is None:
            continue
        yrs, series = ind
        sig = trend_sig(series[indicator], method=method, window=window)
        valid = (yrs >= lo) & (yrs <= hi) & ~np.isnan(series[indicator])
        flagged += int(np.sum(sig & valid))
        total += int(np.sum(valid))
    return flagged, total


def main():
    print("=" * 70)
    print("CSD FIX ATTEMPT: detrended LEVELS (sigma=3), matched detection vs FP")
    print("=" * 70)
    ep = [(n, i["onset"] - lead_for(i), i["onset"] - 1) for n, i in KNOWN_EPISODES.items()]
    st = [(n, BASELINE_END + 1, 2025) for n in STABLE_DEMOCRACIES]
    out = {}
    for ind in ["dom_eig", "var", "ar1", "total_var"]:
        dfl, dto = yearrate(ep, ind)
        ffl, fto = yearrate(st, ind)
        dr, fr = dfl / max(dto, 1), ffl / max(fto, 1)
        flag = "PASS (detection > FP)" if dr > fr else "no separation"
        print(f"  {ind:10s}: pre-onset {dfl}/{dto}={dr:.0%}  stable-FP {ffl}/{fto}={fr:.0%}  -> {flag}")
        out[f"{ind}_detect"] = dr
        out[f"{ind}_fp"] = fr
    print("\n  [differenced-residual baseline (csd_hardening): pre-onset 10% vs FP 11%]")
    pd.DataFrame([out]).to_csv(os.path.join(OUTPUT_DIR, "csd_levels_fix_results.csv"), index=False)
    print(f"\nSaved to robustness/csd_levels_fix_results.csv")


if __name__ == "__main__":
    main()
