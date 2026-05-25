"""Hardening the critical-slowing-down (CSD) early-warning component against the
standard EWS critiques (Boettiger-Hastings prosecutor's fallacy; Dakos et al.
surrogate testing; Deb et al. short-series power).

  detection_vs_fp : CSD detection rate in episode pre-onset windows vs the
                    false-positive rate on stable democracies (the prosecutor's-
                    fallacy defense -- detection must beat the FP rate).
  by_type         : detection split by gradual backsliding vs coup (CSD is
                    expected to fire for gradual onsets, not noise-driven coups).
  three_surrogate : ARMA(1) (production), phase-randomized, and bootstrap
                    surrogates -- agreement across nulls.
  per_indicator   : which indicator (dominant eigenvalue, cross-correlation,
                    total variance, variance, AR1) carries the signal.
  window_sensitivity : detection rate across rolling-window sizes.

Reuses load_residuals, multivariate_csd, rolling_stats from stage5_ews.estimate
so indicators are computed exactly as in production.
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
from scipy import stats

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from stage5_ews.estimate import (load_residuals, multivariate_csd, rolling_stats,
                                  KNOWN_EPISODES, lead_for, WINDOW, MIN_WINDOW,
                                  BASELINE_END, KENDALL_SIG)
from false_positive_analysis import STABLE_DEMOCRACIES

OUTPUT_DIR = os.path.dirname(os.path.abspath(__file__))
FACTOR_RESID = ["resid_factor_1", "resid_factor_2", "resid_factor_3", "resid_factor_4"]
N_SURR = 40
RNG = np.random.default_rng(42)
RESID = None


def gen_surrogate(c, method):
    n = len(c)
    c = np.asarray(c, float)
    sd = np.std(c)
    if method == "arma":
        if sd < 1e-10:
            return c.copy()
        ar1 = np.clip(np.corrcoef(c[:-1], c[1:])[0, 1], -0.99, 0.99)
        rstd = sd * np.sqrt(max(1 - ar1 ** 2, 1e-12))
        s = np.empty(n)
        s[0] = RNG.normal(0, sd)
        for i in range(1, n):
            s[i] = ar1 * s[i - 1] + RNG.normal(0, max(rstd, 1e-10))
        return s
    if method == "phase":
        x = c - c.mean()
        F = np.fft.rfft(x)
        ph = RNG.uniform(0, 2 * np.pi, len(F))
        ph[0] = 0.0
        if n % 2 == 0:
            ph[-1] = 0.0
        return np.fft.irfft(np.abs(F) * np.exp(1j * ph), n=n)
    return RNG.permutation(c)


def trend_sig(series, method="arma", window=WINDOW, n_surr=N_SURR, sig=KENDALL_SIG):
    n = len(series)
    out = np.zeros(n, dtype=bool)
    for t in range(MIN_WINDOW, n):
        c = series[max(0, t - window):t + 1]
        c = c[~np.isnan(c)]
        if len(c) < 5:
            continue
        tau, _ = stats.kendalltau(np.arange(len(c)), c)
        if not (tau > 0):
            continue
        surr = np.array([stats.kendalltau(np.arange(len(c)), gen_surrogate(c, method))[0]
                         for _ in range(n_surr)])
        if np.mean(surr >= tau) < sig:
            out[t] = True
    return out


def indicators(name, window=WINDOW):
    sub = RESID[RESID["country_name"] == name].sort_values("year")
    if len(sub) < MIN_WINDOW + 3:
        return None
    yrs = sub["year"].values
    M = sub[FACTOR_RESID].values
    dom_eig, mean_xcorr, total_var = multivariate_csd(M, window=window)
    stat = [rolling_stats(M[:, k], window=window) for k in range(M.shape[1])]
    var = np.nanmean([s[0] for s in stat], axis=0)
    ar1 = np.nanmean([s[1] for s in stat], axis=0)
    return yrs, {"dom_eig": dom_eig, "mean_xcorr": mean_xcorr, "total_var": total_var,
                 "var": var, "ar1": ar1}


def hit_in_window(yrs, series, lo, hi, method="arma", window=WINDOW):
    sig = trend_sig(series, method=method, window=window)
    mask = (yrs >= lo) & (yrs <= hi)
    return bool(np.any(sig[mask]))


def episode_rate(indicator="dom_eig", method="arma", window=WINDOW, subset=None):
    hits = total = 0
    for name, info in KNOWN_EPISODES.items():
        if subset and info.get("type") != subset:
            continue
        ind = indicators(name, window=window)
        if ind is None:
            continue
        yrs, series = ind
        lo, hi = info["onset"] - lead_for(info), info["onset"] - 1
        total += 1
        hits += hit_in_window(yrs, series[indicator], lo, hi, method=method, window=window)
    return hits, total


def yearrate(pairs, indicator="dom_eig", method="arma", window=WINDOW):
    """Per-country-year significance rate over the given (name, lo, hi) windows,
    counting only years where the indicator is defined -- matched exposure so the
    pre-onset detection rate and the stable false-positive rate are comparable."""
    flagged = total = 0
    for name, lo, hi in pairs:
        ind = indicators(name, window=window)
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
    print("CSD HARDENING")
    print("=" * 70)
    global RESID
    RESID, _ = load_residuals()
    out = {}

    print("\n[A] Per-country-year significance rate (matched exposure): pre-onset vs stable")
    ep_pairs = [(n, i["onset"] - lead_for(i), i["onset"] - 1) for n, i in KNOWN_EPISODES.items()]
    st_pairs = [(n, BASELINE_END + 1, 2025) for n in STABLE_DEMOCRACIES]
    dfl, dto = yearrate(ep_pairs, "dom_eig", "arma")
    ffl, fto = yearrate(st_pairs, "dom_eig", "arma")
    dh, dt = episode_rate("dom_eig", "arma")
    print(f"    pre-onset years flagged   : {dfl}/{dto} = {dfl/max(dto,1):.0%}")
    print(f"    stable years flagged (FP) : {ffl}/{fto} = {ffl/max(fto,1):.0%}")
    print(f"    per-episode detection (any flag in window): {dh}/{dt} = {dh/dt:.0%}")
    verdict = "PASS: pre-onset rate > FP rate" if dfl/max(dto,1) > ffl/max(fto,1) else "WEAK: FP rate >= detection rate"
    print(f"    => {verdict}")
    out.update({"preonset_yearrate": dfl/max(dto,1), "stable_yearrate": ffl/max(fto,1),
                "episode_detect": dh/dt})

    print("\n[B] Detection by episode type (dom_eig, ARMA)")
    bh, bt = episode_rate("dom_eig", "arma", subset="backsliding")
    ch, ct = episode_rate("dom_eig", "arma", subset="coup")
    print(f"    gradual backsliding: {bh}/{bt} = {bh/max(bt,1):.0%}")
    print(f"    coup / sudden      : {ch}/{ct} = {ch/max(ct,1):.0%}  (CSD expected weaker)")
    out.update({"detect_backsliding": bh / max(bt, 1), "detect_coup": ch / max(ct, 1)})

    print("\n[C] Three-surrogate comparison (episode detection, dom_eig)")
    for method in ["arma", "phase", "bootstrap"]:
        h, t = episode_rate("dom_eig", method)
        print(f"    {method:10s}: {h}/{t} = {h/max(t,1):.0%}")
        out[f"detect_{method}"] = h / max(t, 1)

    print("\n[D] Per-indicator detection (episodes, ARMA)")
    for ind in ["dom_eig", "mean_xcorr", "total_var", "var", "ar1"]:
        h, t = episode_rate(ind, "arma")
        print(f"    {ind:11s}: {h}/{t} = {h/max(t,1):.0%}")
        out[f"detect_indicator_{ind}"] = h / max(t, 1)

    print("\n[E] Window sensitivity (episode detection, dom_eig, ARMA)")
    for w in [6, 8, 10, 12]:
        h, t = episode_rate("dom_eig", "arma", window=w)
        print(f"    window={w:2d}: {h}/{t} = {h/max(t,1):.0%}")
        out[f"detect_window_{w}"] = h / max(t, 1)

    pd.DataFrame([out]).to_csv(os.path.join(OUTPUT_DIR, "csd_hardening_results.csv"), index=False)
    print(f"\nSaved to robustness/csd_hardening_results.csv")
    print("Note: ~50-point annual series are below the Kendall-tau power floor "
          "(Deb et al. 2022); surrogate p-values are the appropriate null given length.")


if __name__ == "__main__":
    main()
