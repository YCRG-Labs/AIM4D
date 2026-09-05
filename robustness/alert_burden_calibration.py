"""
False-alert rates, watchlist burden, and calibration diagnostics.

Two referee points motivate this. First, "zero stable-democracy false
positives" is reported without a denominator or a time window, which makes it
unfalsifiable; what an operator needs is the annual alert rate over the whole
eligible risk set and the resulting watchlist size. Second, the reliability
diagram's top bin holds six country-years, so the curve carries no usable
uncertainty; calibration slope and intercept with bootstrap intervals, and
equal-frequency bins, are the standard summaries.

Outputs robustness/alert_burden.csv and robustness/calibration_diagnostics.csv.
"""

import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

OUT = os.path.dirname(os.path.abspath(__file__))
TRAIN_CUTOFF = 2019
TIERS = {"watch": 0.80, "warning": 0.95, "alert": 0.98}
STABLE = ["Denmark", "Sweden", "Norway", "Switzerland", "Finland", "Germany",
          "Canada", "New Zealand", "Australia", "Ireland"]
N_BOOT = 2000
RNG = np.random.default_rng(42)


def load():
    d = pd.read_csv(os.path.join(OUT, "..", "stage5_ews", "ews_signals.csv"))
    d = d.dropna(subset=["label", "calibrated_risk"])
    if "is_postonset" in d.columns:
        d = d[~d["is_postonset"].fillna(False)]
    return d


def tier_thresholds(d):
    train_neg = d[(d.year <= TRAIN_CUTOFF) & (d.label == 0)]["calibrated_risk"]
    return {k: float(train_neg.quantile(q)) for k, q in TIERS.items()}


def alert_burden(d, thr):
    rows = []
    oos = d[d.year > TRAIN_CUTOFF]
    for scope_name, scope in [("all eligible country-years", d),
                              ("out-of-sample 2020-2025", oos),
                              ("consolidated democracies", d[d.country_name.isin(STABLE)]),
                              ("consolidated democracies, 2020-2025",
                               oos[oos.country_name.isin(STABLE)])]:
        neg = scope[scope.label == 0]
        n_years = scope.year.nunique()
        for tier, t in thr.items():
            fired = int((neg["calibrated_risk"] >= t).sum())
            n = len(neg)
            rows.append({
                "scope": scope_name,
                "tier": tier,
                "threshold": round(t, 4),
                "negative_country_years": n,
                "false_alerts": fired,
                "false_alert_rate": round(fired / n, 4) if n else np.nan,
                "years_covered": n_years,
                "mean_countries_flagged_per_year": round(fired / n_years, 2) if n_years else np.nan,
            })
    return pd.DataFrame(rows)


def calibration(d):
    oos = d[d.year > TRAIN_CUTOFF]
    p = oos["calibrated_risk"].to_numpy(float)
    y = oos["label"].to_numpy(int)
    eps = 1e-6
    logit = np.log(np.clip(p, eps, 1 - eps) / (1 - np.clip(p, eps, 1 - eps)))

    def fit(idx):
        from sklearn.linear_model import LogisticRegression
        m = LogisticRegression(penalty=None, solver="lbfgs", max_iter=2000)
        m.fit(logit[idx].reshape(-1, 1), y[idx])
        return float(m.coef_[0][0]), float(m.intercept_[0])

    slope, inter = fit(np.arange(len(y)))
    countries = oos["country_name"].to_numpy()
    uniq = np.unique(countries)
    bs, bi = [], []
    for _ in range(N_BOOT):
        draw = RNG.choice(uniq, size=len(uniq), replace=True)
        idx = np.concatenate([np.where(countries == c)[0] for c in draw])
        if y[idx].sum() < 3 or y[idx].sum() == len(idx):
            continue
        try:
            s, i = fit(idx)
            bs.append(s); bi.append(i)
        except Exception:
            continue
    q = lambda a: (float(np.percentile(a, 2.5)), float(np.percentile(a, 97.5)))
    s_lo, s_hi = q(bs); i_lo, i_hi = q(bi)

    rows = [{"quantity": "calibration slope", "estimate": round(slope, 4),
             "ci_low": round(s_lo, 4), "ci_high": round(s_hi, 4),
             "ideal": 1.0, "n_boot": len(bs)},
            {"quantity": "calibration intercept", "estimate": round(inter, 4),
             "ci_low": round(i_lo, 4), "ci_high": round(i_hi, 4),
             "ideal": 0.0, "n_boot": len(bi)}]

    dec = pd.qcut(p, 5, duplicates="drop")
    g = pd.DataFrame({"p": p, "y": y, "bin": dec}).groupby("bin", observed=True)
    for b, sub in g:
        rows.append({"quantity": f"equal-frequency bin {b}",
                     "estimate": round(sub.p.mean(), 4),
                     "ci_low": round(sub.y.mean(), 4), "ci_high": np.nan,
                     "ideal": np.nan, "n_boot": len(sub)})
    return pd.DataFrame(rows), slope, inter, (s_lo, s_hi), (i_lo, i_hi)


def main():
    d = load()
    thr = tier_thresholds(d)
    print("tier thresholds (training-negative percentiles):")
    for k, v in thr.items():
        print(f"  {k:<8} {v:.4f}")
    print()

    ab = alert_burden(d, thr)
    ab.to_csv(os.path.join(OUT, "alert_burden.csv"), index=False)
    print(ab.to_string(index=False))
    print()

    cal, slope, inter, sci, ici = calibration(d)
    cal.to_csv(os.path.join(OUT, "calibration_diagnostics.csv"), index=False)
    print(f"calibration slope     {slope:.3f}  95% CI [{sci[0]:.3f}, {sci[1]:.3f}]  (ideal 1)")
    print(f"calibration intercept {inter:.3f}  95% CI [{ici[0]:.3f}, {ici[1]:.3f}]  (ideal 0)")
    print()
    print(cal[cal.quantity.str.startswith("equal-frequency")].to_string(index=False))
    print(f"\nWrote alert_burden.csv and calibration_diagnostics.csv")


if __name__ == "__main__":
    main()
