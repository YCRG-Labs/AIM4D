"""
Bootstrap 95% confidence intervals for the paper's headline metrics.

Loads stage5_ews/ews_signals.csv (must have combined_risk + label populated)
and V-Dem v16 for the FH/Polity outcome cross-checks. Resamples 1,000x with
replacement; reports point estimate and percentile [2.5, 97.5] CI.
Outputs robustness/bootstrap_cis.csv.
"""

import os
import sys
import warnings

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score, roc_auc_score, brier_score_loss, log_loss,
)


EPS = 1e-15

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

OUT = os.path.dirname(os.path.abspath(__file__))
RNG = np.random.default_rng(42)
N_BOOT = 2000


def _percentile_ci(arr, alpha=0.05):
    return float(np.percentile(arr, 100 * alpha / 2)), float(np.percentile(arr, 100 * (1 - alpha / 2)))


def _bca_ci(arr, theta_hat, jack_stats, alpha=0.05):
    """
    Bias-corrected and accelerated (BCa) interval (Efron 1987).
    Falls back to percentile if BCa edge-cases hit (degenerate z0 or a).
    """
    from scipy.stats import norm
    arr = np.asarray(arr)
    p_below = float(np.mean(arr < theta_hat))
    if p_below in (0.0, 1.0):
        return _percentile_ci(arr, alpha)
    z0 = float(norm.ppf(p_below))
    jack_stats = np.asarray(jack_stats)
    if len(jack_stats) < 2:
        return _percentile_ci(arr, alpha)
    jack_mean = float(jack_stats.mean())
    diffs = jack_mean - jack_stats
    num = float(np.sum(diffs ** 3))
    den = 6.0 * float(np.sum(diffs ** 2)) ** 1.5
    if den == 0:
        return _percentile_ci(arr, alpha)
    a = num / den
    z_lo = float(norm.ppf(alpha / 2))
    z_hi = float(norm.ppf(1 - alpha / 2))
    def _adj(z):
        denom = 1.0 - a * (z0 + z)
        if abs(denom) < 1e-6:
            return float(norm.cdf(z0 + z))
        return float(norm.cdf(z0 + (z0 + z) / denom))
    alpha1 = max(1e-4, min(1 - 1e-4, _adj(z_lo)))
    alpha2 = max(1e-4, min(1 - 1e-4, _adj(z_hi)))
    return float(np.percentile(arr, 100 * alpha1)), float(np.percentile(arr, 100 * alpha2))


def bootstrap_auc(y, s, fn=roc_auc_score, n_boot=N_BOOT, clusters=None, method="bca"):
    """
    Bootstrap CI. With `clusters`, uses pairs cluster bootstrap (Cameron-Miller
    2015). With method="bca" (default), uses bias-corrected and accelerated
    intervals (Efron 1987) — typically 10-30% tighter than percentile at small N.
    method="percentile" gives the older simple percentile interval.

    Returns (theta_hat_on_full_data, ci_low, ci_high).
    """
    y = np.asarray(y)
    s = np.asarray(s)

    try:
        theta_hat = float(fn(y, s))
    except Exception:
        return np.nan, np.nan, np.nan

    boot_stats = []
    if clusters is not None:
        clusters = np.asarray(clusters)
        unique = np.unique(clusters)
        by_cluster = {c: np.where(clusters == c)[0] for c in unique}
        for _ in range(n_boot):
            draw = RNG.choice(unique, size=len(unique), replace=True)
            idx = np.concatenate([by_cluster[c] for c in draw])
            yy, ss = y[idx], s[idx]
            if yy.sum() < 2 or yy.sum() == len(yy):
                continue
            try:
                boot_stats.append(fn(yy, ss))
            except Exception:
                continue
    else:
        n = len(y)
        unique = None
        by_cluster = None
        for _ in range(n_boot):
            idx = RNG.integers(0, n, n)
            yy, ss = y[idx], s[idx]
            if yy.sum() == 0 or yy.sum() == len(yy):
                continue
            try:
                boot_stats.append(fn(yy, ss))
            except Exception:
                continue
    if not boot_stats:
        return theta_hat, np.nan, np.nan
    arr = np.asarray(boot_stats)

    if method == "percentile":
        lo, hi = _percentile_ci(arr)
        return theta_hat, lo, hi

    jack_stats = []
    if clusters is not None:
        for c in unique:
            idx = np.where(clusters != c)[0]
            yy, ss = y[idx], s[idx]
            if yy.sum() < 2 or yy.sum() == len(yy):
                continue
            try:
                jack_stats.append(fn(yy, ss))
            except Exception:
                continue
    else:
        n = len(y)
        sample_idx = np.arange(n) if n <= 500 else RNG.choice(n, size=500, replace=False)
        for i in sample_idx:
            idx = np.delete(np.arange(n), i)
            yy, ss = y[idx], s[idx]
            if yy.sum() < 2 or yy.sum() == len(yy):
                continue
            try:
                jack_stats.append(fn(yy, ss))
            except Exception:
                continue
    lo, hi = _bca_ci(arr, theta_hat, jack_stats)
    return theta_hat, lo, hi


def wilson_ci(k, n, alpha=0.05):
    if n == 0:
        return 0.0, 0.0
    from scipy.stats import beta
    lo = beta.ppf(alpha / 2, k, n - k + 1) if k > 0 else 0.0
    hi = beta.ppf(1 - alpha / 2, k + 1, n - k) if k < n else 1.0
    return float(lo), float(hi)


def main():
    rows = []

    print("=" * 70)
    print("BOOTSTRAP 95% CIs ON HEADLINE METRICS")
    print("=" * 70)

    ews_path = os.path.join(OUT, "..", "stage5_ews", "ews_signals.csv")
    ews = pd.read_csv(ews_path)
    if "combined_risk" not in ews.columns:
        raise RuntimeError("ews_signals.csv missing combined_risk column; rerun stage 5")
    if "label" not in ews.columns:
        raise RuntimeError("ews_signals.csv missing label column")

    valid = ews.dropna(subset=["combined_risk", "label"])
    if "is_postonset" in valid.columns:
        valid = valid[~valid["is_postonset"]].copy()
    y = valid["label"].astype(int).values
    s = valid["combined_risk"].values
    clusters = valid["country_name"].values

    print(f"\nFull panel (post-onset excluded): n={len(valid)}, "
          f"n_positive={int(y.sum())} ({y.mean():.1%}), "
          f"n_countries={len(np.unique(clusters))}\n")

    auc, lo, hi = bootstrap_auc(y, s, roc_auc_score, clusters=clusters)
    print(f"  AUC-ROC (in-sample, BCa cluster):  {auc:.3f}  95% CI [{lo:.3f}, {hi:.3f}]")
    rows.append({"metric": "auc_roc_in_sample", "point": auc, "ci_low": lo, "ci_high": hi, "n": len(y)})

    auc_pr, lo, hi = bootstrap_auc(y, s, average_precision_score, clusters=clusters)
    print(f"  AUC-PR  (in-sample, BCa cluster):  {auc_pr:.3f}  95% CI [{lo:.3f}, {hi:.3f}]")
    rows.append({"metric": "auc_pr_in_sample", "point": auc_pr, "ci_low": lo, "ci_high": hi, "n": len(y)})

    def _brier(yy, ss, ww=None):
        return brier_score_loss(yy, np.clip(ss, EPS, 1 - EPS), sample_weight=ww)
    def _logloss(yy, ss, ww=None):
        return log_loss(yy, np.clip(ss, EPS, 1 - EPS),
                        sample_weight=ww, labels=[0, 1])
    def _bss(yy, ss, ww=None):
        p_clim = np.average(yy, weights=ww) if ww is not None else float(yy.mean())
        ref = np.average((yy - p_clim) ** 2, weights=ww) if ww is not None else float(((yy - p_clim) ** 2).mean())
        return 1.0 - _brier(yy, ss, ww) / max(ref, EPS)

    brier_pt, lo, hi = bootstrap_auc(y, s, _brier, clusters=clusters)
    print(f"  Brier   (in-sample, BCa cluster):  {brier_pt:.4f}  95% CI [{lo:.4f}, {hi:.4f}]")
    rows.append({"metric": "brier_in_sample", "point": brier_pt, "ci_low": lo, "ci_high": hi, "n": len(y)})

    ll_pt, lo, hi = bootstrap_auc(y, s, _logloss, clusters=clusters)
    print(f"  LogLoss (in-sample, BCa cluster):  {ll_pt:.4f}  95% CI [{lo:.4f}, {hi:.4f}]")
    rows.append({"metric": "logloss_in_sample", "point": ll_pt, "ci_low": lo, "ci_high": hi, "n": len(y)})

    bss_pt, lo, hi = bootstrap_auc(y, s, _bss, clusters=clusters)
    print(f"  BSS     (in-sample, BCa cluster):  {bss_pt:.4f}  95% CI [{lo:.4f}, {hi:.4f}]")
    rows.append({"metric": "bss_in_sample", "point": bss_pt, "ci_low": lo, "ci_high": hi, "n": len(y)})

    cutoff = 2019
    oos = valid[valid["year"] > cutoff]
    if len(oos) > 0 and oos["label"].sum() > 1:
        auc, lo, hi = bootstrap_auc(
            oos["label"].astype(int).values,
            oos["combined_risk"].values,
            roc_auc_score,
            clusters=oos["country_name"].values,
        )
        print(f"  AUC-ROC (OOS year>{cutoff}, BCa cluster): {auc:.3f}  95% CI [{lo:.3f}, {hi:.3f}]  n={len(oos)}")
        rows.append({"metric": f"auc_roc_oos_{cutoff}", "point": auc, "ci_low": lo, "ci_high": hi, "n": len(oos)})

        auc_pr, lo, hi = bootstrap_auc(
            oos["label"].astype(int).values,
            oos["combined_risk"].values,
            average_precision_score,
            clusters=oos["country_name"].values,
        )
        print(f"  AUC-PR  (OOS year>{cutoff}, BCa cluster): {auc_pr:.3f}  95% CI [{lo:.3f}, {hi:.3f}]  n={len(oos)}")
        rows.append({"metric": f"auc_pr_oos_{cutoff}", "point": auc_pr, "ci_low": lo, "ci_high": hi, "n": len(oos)})

        y_oos = oos["label"].astype(int).values
        s_oos = oos["combined_risk"].values
        c_oos = oos["country_name"].values
        for fn_name, fn in [("brier", _brier), ("logloss", _logloss), ("bss", _bss)]:
            pt, lo_, hi_ = bootstrap_auc(y_oos, s_oos, fn, clusters=c_oos)
            print(f"  {fn_name:8s} (OOS year>{cutoff}, BCa cluster): {pt:.4f}  95% CI [{lo_:.4f}, {hi_:.4f}]  n={len(oos)}")
            rows.append({"metric": f"{fn_name}_oos_{cutoff}", "point": pt, "ci_low": lo_, "ci_high": hi_, "n": len(oos)})

    try:
        from stage5_ews.estimate import KNOWN_EPISODES, LEAD_YEARS
    except Exception:
        LEAD_YEARS = 5
        KNOWN_EPISODES = {}

    if KNOWN_EPISODES:
        episodes = []
        for country, info in KNOWN_EPISODES.items():
            onset = info["onset"]
            sub = valid[(valid["country_name"] == country)
                        & (valid["year"] >= onset - LEAD_YEARS)
                        & (valid["year"] < onset)]
            if len(sub) == 0:
                continue
            max_risk = sub["combined_risk"].max()
            train_risks = valid[valid["year"] <= 2021]["combined_risk"]
            thresh = train_risks.quantile(0.80)
            episodes.append({
                "country": country,
                "type": info["type"],
                "max_risk": float(max_risk),
                "detected": int(max_risk >= thresh),
            })

        if episodes:
            ep_df = pd.DataFrame(episodes)
            for stratum_name, mask in [
                ("in_sample_watch_all", np.ones(len(ep_df), dtype=bool)),
                ("in_sample_watch_backsliding", ep_df["type"].eq("backsliding").values),
                ("in_sample_watch_coup", ep_df["type"].eq("coup").values),
            ]:
                if mask.sum() == 0:
                    continue
                hits = int(ep_df[mask]["detected"].sum())
                n = int(mask.sum())
                point = hits / n
                lo, hi = wilson_ci(hits, n)
                print(f"  {stratum_name:<28}: {hits}/{n} ({point:.0%})  95% CI [{lo:.0%}, {hi:.0%}]")
                rows.append({"metric": stratum_name, "point": point, "ci_low": lo, "ci_high": hi, "n": n})

    loeo_path = os.path.join(OUT, "..", "stage5_ews", "loeo_results.csv")
    if os.path.exists(loeo_path):
        loeo = pd.read_csv(loeo_path)
        loeo_strata = [
            ("loeo_watch_all", loeo["detected_watch"], np.ones(len(loeo), dtype=bool)),
            ("loeo_watch_backsliding", loeo["detected_watch"], loeo["type"].eq("backsliding").values),
            ("loeo_watch_coup", loeo["detected_watch"], loeo["type"].eq("coup").values),
        ]
        for stratum_name, det_col, mask in loeo_strata:
            if mask.sum() == 0:
                continue
            hits = int(det_col[mask].sum())
            n = int(mask.sum())
            point = hits / n
            lo, hi = wilson_ci(hits, n)
            print(f"  {stratum_name:<28}: {hits}/{n} ({point:.0%})  95% CI [{lo:.0%}, {hi:.0%}]")
            rows.append({"metric": stratum_name, "point": point, "ci_low": lo, "ci_high": hi, "n": n})
    else:
        print(f"\n  [note] {loeo_path} not present — true LOEO CIs skipped.")
        print(f"         Rerun stage 5 after the patch below to dump LOEO results.")

    vdem_path = os.path.join(OUT, "..", "data", "vdem_v16.csv")
    fh_cols = ["country_text_id", "year", "e_fh_pr"]
    fh = pd.read_csv(vdem_path, usecols=fh_cols, low_memory=False).dropna()
    fh["e_fh_pr"] = fh["e_fh_pr"].astype(float)
    fh = fh.sort_values(["country_text_id", "year"])
    fh["fh_3yr_change"] = fh.groupby("country_text_id")["e_fh_pr"].diff(3)
    fh["fh_decline_3yr_2pt"] = (fh["fh_3yr_change"] >= 2).astype(int)

    merged_fh = valid.merge(fh, on=["country_text_id", "year"], how="inner").dropna(subset=["fh_decline_3yr_2pt"])
    if merged_fh["fh_decline_3yr_2pt"].sum() > 1:
        auc, lo, hi = bootstrap_auc(merged_fh["fh_decline_3yr_2pt"].astype(int).values,
                                     merged_fh["combined_risk"].values, roc_auc_score)
        print(f"  FH 3yr decline AUC:   {auc:.3f}  95% CI [{lo:.3f}, {hi:.3f}]  n={len(merged_fh)}")
        rows.append({"metric": "auc_fh_3yr_decline_2pt", "point": auc, "ci_low": lo, "ci_high": hi, "n": len(merged_fh)})

    polity = pd.read_csv(vdem_path, usecols=["country_text_id", "year", "e_polity2"], low_memory=False).dropna()
    polity["e_polity2"] = polity["e_polity2"].astype(float)
    polity = polity.sort_values(["country_text_id", "year"])
    polity["polity_3yr_change"] = polity.groupby("country_text_id")["e_polity2"].diff(3)
    polity["polity_decline_3yr_3pt"] = (polity["polity_3yr_change"] <= -3).astype(int)

    merged_polity = valid.merge(polity, on=["country_text_id", "year"], how="inner").dropna(subset=["polity_decline_3yr_3pt"])
    if merged_polity["polity_decline_3yr_3pt"].sum() > 1:
        auc, lo, hi = bootstrap_auc(merged_polity["polity_decline_3yr_3pt"].astype(int).values,
                                     merged_polity["combined_risk"].values, roc_auc_score)
        print(f"  Polity 3yr decline:   {auc:.3f}  95% CI [{lo:.3f}, {hi:.3f}]  n={len(merged_polity)}")
        rows.append({"metric": "auc_polity_3yr_decline_3pt", "point": auc, "ci_low": lo, "ci_high": hi, "n": len(merged_polity)})

    out = pd.DataFrame(rows)
    out_path = os.path.join(OUT, "bootstrap_cis.csv")
    out.to_csv(out_path, index=False)
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
