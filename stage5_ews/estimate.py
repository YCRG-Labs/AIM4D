import sys
import os
import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

WINDOW = 8
MIN_WINDOW = 5
BASELINE_END = 2005
TRAIN_CUTOFF = 2019
Z_THRESHOLD = 1.5
PERSISTENCE = 2
LEAD_YEARS = 5
MIN_ABS_VAR_PERCENTILE = 0.30
Z_CAP = 10.0

KNOWN_EPISODES = {
    "Hungary": {"onset": 2010, "peak": 2018, "type": "backsliding"},
    "Türkiye": {"onset": 2013, "peak": 2017, "type": "backsliding"},
    "Poland": {"onset": 2015, "peak": 2019, "type": "backsliding"},
    "Venezuela": {"onset": 2005, "peak": 2013, "type": "backsliding"},
    "Tunisia": {"onset": 2021, "peak": 2023, "type": "backsliding"},
    "Burma/Myanmar": {"onset": 2021, "peak": 2022, "type": "coup"},
    "Mali": {"onset": 2020, "peak": 2021, "type": "coup"},
    "Burkina Faso": {"onset": 2022, "peak": 2022, "type": "coup"},
}


def load_residuals():
    base = os.path.dirname(os.path.abspath(__file__))
    scores = pd.read_csv(os.path.join(base, "..", "stage4_nscm", "contagion_scores.csv"))
    factors = pd.read_csv(os.path.join(base, "..", "stage1_factors", "country_year_factors.csv"))

    merged = factors.merge(
        scores[["country_text_id", "year", "contagion_score", "domestic_score"]],
        on=["country_text_id", "year"], how="inner"
    )

    factor_cols = ["factor_1", "factor_2", "factor_3", "factor_4"]
    for k, fc in enumerate(factor_cols):
        merged[f"resid_{k+1}"] = merged.groupby("country_text_id")[fc].diff() * merged["domestic_score"]

    merged = merged.dropna(subset=[f"resid_{k+1}" for k in range(4)])
    return merged


def compute_rolling_stats(series, window=WINDOW, min_window=MIN_WINDOW):
    n = len(series)
    rolling_var = np.full(n, np.nan)
    rolling_ar1 = np.full(n, np.nan)

    for t in range(min_window, n):
        start = max(0, t - window)
        chunk = series[start:t + 1]
        if len(chunk) < min_window:
            continue
        rolling_var[t] = np.var(chunk, ddof=1) if len(chunk) > 1 else 0
        if len(chunk) >= 3 and np.std(chunk) > 1e-10:
            rolling_ar1[t] = np.corrcoef(chunk[:-1], chunk[1:])[0, 1]

    return rolling_var, rolling_ar1


def country_relative_zscore(values, years, baseline_end=BASELINE_END):
    baseline_mask = np.array(years) <= baseline_end
    valid_baseline = values[baseline_mask & ~np.isnan(values)]

    if len(valid_baseline) < 3:
        valid_all = values[~np.isnan(values)]
        if len(valid_all) < 3:
            return np.full(len(values), np.nan)
        mu, sigma = np.mean(valid_all), np.std(valid_all)
    else:
        mu, sigma = np.mean(valid_baseline), np.std(valid_baseline)

    if sigma < 1e-10:
        sigma = np.std(values[~np.isnan(values)])
    if sigma < 1e-10:
        return np.full(len(values), 0.0)

    return (values - mu) / sigma


def rolling_kendall(values, window=8):
    n = len(values)
    taus = np.full(n, np.nan)
    for t in range(5, n):
        start = max(0, t - window)
        chunk = values[start:t + 1]
        valid = ~np.isnan(chunk)
        if valid.sum() >= 4:
            tau, _ = stats.kendalltau(np.arange(valid.sum()), chunk[valid])
            taus[t] = tau
    return taus


def persistence_filter(alerts, min_consecutive=PERSISTENCE):
    filtered = np.zeros(len(alerts), dtype=bool)
    count = 0
    for i in range(len(alerts)):
        if alerts[i]:
            count += 1
            if count >= min_consecutive:
                filtered[i] = True
                if count == min_consecutive:
                    for j in range(max(0, i - min_consecutive + 1), i):
                        filtered[j] = True
        else:
            count = 0
    return filtered


def run_ews():
    print("=== Stage 5: Early Warning Signals (Critical Slowing Down) ===\n")

    df = load_residuals()
    countries = sorted(df["country_name"].unique())
    resid_cols = [f"resid_{k+1}" for k in range(4)]
    print(f"Countries: {len(countries)}")
    print(f"Per-factor residuals: {resid_cols}")
    print(f"Baseline period: ≤{BASELINE_END}")
    print(f"Country-relative z-score threshold: {Z_THRESHOLD}")
    print(f"Persistence filter: {PERSISTENCE} consecutive years")

    all_rolling_vars = []
    for country in countries:
        cdf = df[df["country_name"] == country].sort_values("year")
        if len(cdf) < MIN_WINDOW + 2:
            continue
        for rcol in resid_cols:
            rv, _ = compute_rolling_stats(cdf[rcol].values)
            train_rv = rv[np.array(cdf["year"].values) <= TRAIN_CUTOFF]
            all_rolling_vars.extend(train_rv[~np.isnan(train_rv)])

    abs_var_floor = np.percentile(all_rolling_vars, MIN_ABS_VAR_PERCENTILE * 100)
    print(f"Absolute variance floor (p{int(MIN_ABS_VAR_PERCENTILE*100)} of train): {abs_var_floor:.6f}")

    all_ews = []

    for country in countries:
        cdf = df[df["country_name"] == country].sort_values("year")
        if len(cdf) < MIN_WINDOW + 2:
            continue

        years = cdf["year"].values
        country_id = cdf["country_text_id"].iloc[0]

        factor_alerts = np.zeros(len(years), dtype=int)
        best_var_z = np.full(len(years), np.nan)
        best_ar1_z = np.full(len(years), np.nan)
        best_var_tau = np.full(len(years), np.nan)
        best_ar1_tau = np.full(len(years), np.nan)
        max_abs_var = np.full(len(years), np.nan)

        for rcol in resid_cols:
            series = cdf[rcol].values
            rolling_var, rolling_ar1 = compute_rolling_stats(series)

            var_z = np.clip(country_relative_zscore(rolling_var, years), -Z_CAP, Z_CAP)
            ar1_z = np.clip(country_relative_zscore(rolling_ar1, years), -Z_CAP, Z_CAP)

            var_tau = rolling_kendall(rolling_var)
            ar1_tau = rolling_kendall(rolling_ar1)

            for t in range(len(years)):
                above_floor = not np.isnan(rolling_var[t]) and rolling_var[t] > abs_var_floor

                if not np.isnan(var_z[t]) and not np.isnan(ar1_z[t]) and above_floor:
                    csd_signal = (
                        (var_z[t] > Z_THRESHOLD and ar1_z[t] > Z_THRESHOLD) or
                        (var_z[t] > Z_THRESHOLD and not np.isnan(ar1_tau[t]) and ar1_tau[t] > 0.3) or
                        (ar1_z[t] > Z_THRESHOLD and not np.isnan(var_tau[t]) and var_tau[t] > 0.3)
                    )
                    if csd_signal:
                        factor_alerts[t] += 1

                if np.isnan(best_var_z[t]) or (not np.isnan(var_z[t]) and var_z[t] > best_var_z[t]):
                    best_var_z[t] = var_z[t]
                if np.isnan(best_ar1_z[t]) or (not np.isnan(ar1_z[t]) and ar1_z[t] > best_ar1_z[t]):
                    best_ar1_z[t] = ar1_z[t]
                if np.isnan(max_abs_var[t]) or (not np.isnan(rolling_var[t]) and rolling_var[t] > max_abs_var[t]):
                    max_abs_var[t] = rolling_var[t]
                if np.isnan(best_var_tau[t]) or (not np.isnan(var_tau[t]) and var_tau[t] > best_var_tau[t]):
                    best_var_tau[t] = var_tau[t]
                if np.isnan(best_ar1_tau[t]) or (not np.isnan(ar1_tau[t]) and ar1_tau[t] > best_ar1_tau[t]):
                    best_ar1_tau[t] = ar1_tau[t]

        csd_index = np.zeros(len(years))
        for t in range(len(years)):
            above_floor = not np.isnan(max_abs_var[t]) and max_abs_var[t] > abs_var_floor
            if not above_floor:
                csd_index[t] = 0
                continue
            components = []
            if not np.isnan(best_var_z[t]):
                components.append(min(Z_CAP, max(0, best_var_z[t])))
            if not np.isnan(best_ar1_z[t]):
                components.append(min(Z_CAP, max(0, best_ar1_z[t])))
            if not np.isnan(best_var_tau[t]):
                components.append(max(0, best_var_tau[t]) * 2)
            if not np.isnan(best_ar1_tau[t]):
                components.append(max(0, best_ar1_tau[t]) * 2)
            csd_index[t] = np.mean(components) if components else 0

        raw_alert = (factor_alerts >= 2) | ((factor_alerts >= 1) & (csd_index > 3.0))
        persistent_alert = persistence_filter(raw_alert, PERSISTENCE)

        for t in range(len(years)):
            all_ews.append({
                "country_name": country,
                "country_text_id": country_id,
                "year": int(years[t]),
                "var_z": best_var_z[t],
                "ar1_z": best_ar1_z[t],
                "var_trend": best_var_tau[t],
                "ar1_trend": best_ar1_tau[t],
                "n_factor_alerts": factor_alerts[t],
                "raw_alert": raw_alert[t],
                "ews_alert": persistent_alert[t],
                "csd_index": csd_index[t],
            })

    ews_df = pd.DataFrame(all_ews)

    output_dir = os.path.dirname(os.path.abspath(__file__))
    ews_df.to_csv(os.path.join(output_dir, "ews_signals.csv"), index=False)

    print(f"\n{'='*60}")
    print(f"Validation against known episodes")
    print(f"{'='*60}\n")

    hits = 0
    total = 0
    for country, info in KNOWN_EPISODES.items():
        onset = info["onset"]
        pre = ews_df[
            (ews_df["country_name"] == country) &
            (ews_df["year"] >= onset - LEAD_YEARS) &
            (ews_df["year"] < onset)
        ]

        if len(pre) == 0:
            print(f"  {country} ({info['type']} {onset}): NO DATA")
            continue

        total += 1
        any_alert = pre["ews_alert"].any()
        any_raw = pre["raw_alert"].any()
        max_csd = pre["csd_index"].max()

        if any_alert:
            hits += 1
            alert_years = sorted(pre[pre["ews_alert"]]["year"].tolist())
            lead_time = onset - alert_years[0]
            print(f"  {country} ({info['type']} {onset}): DETECTED (persistent)")
            print(f"    First alert: {alert_years[0]} ({lead_time}yr lead), CSD index: {max_csd:.2f}")
        elif any_raw:
            hits += 1
            alert_years = sorted(pre[pre["raw_alert"]]["year"].tolist())
            print(f"  {country} ({info['type']} {onset}): DETECTED (raw, not persistent)")
            print(f"    Alert years: {alert_years}, CSD index: {max_csd:.2f}")
        else:
            max_var_z = pre["var_z"].max()
            max_ar1_z = pre["ar1_z"].max()
            print(f"  {country} ({info['type']} {onset}): MISSED")
            print(f"    Max var z: {max_var_z:.2f}, Max AR1 z: {max_ar1_z:.2f}, CSD index: {max_csd:.2f}")

    sensitivity = hits / total if total > 0 else 0
    print(f"\n  Sensitivity: {hits}/{total} ({sensitivity:.0%})")

    print(f"\n{'='*60}")
    print(f"False positive analysis")
    print(f"{'='*60}\n")

    known_countries = set(KNOWN_EPISODES.keys())
    known_windows = {}
    for c, info in KNOWN_EPISODES.items():
        for y in range(info["onset"] - LEAD_YEARS, info["peak"] + 1):
            known_windows[(c, y)] = True

    all_alerts = ews_df[ews_df["ews_alert"]].copy()
    tp = all_alerts[all_alerts.apply(lambda r: (r["country_name"], r["year"]) in known_windows, axis=1)]
    fp = all_alerts[~all_alerts.index.isin(tp.index)]

    total_cy = len(ews_df)
    alert_rate = len(all_alerts) / total_cy
    fp_rate = len(fp) / total_cy
    precision = len(tp) / len(all_alerts) if len(all_alerts) > 0 else 0

    print(f"  Total country-years: {total_cy}")
    print(f"  Persistent alerts: {len(all_alerts)} ({alert_rate:.1%})")
    print(f"  True positives: {len(tp)}")
    print(f"  False positives: {len(fp)} ({fp_rate:.1%} FP rate)")
    print(f"  Precision: {precision:.1%}")
    if sensitivity > 0 and precision > 0:
        f1 = 2 * precision * sensitivity / (precision + sensitivity)
        print(f"  F1 score: {f1:.3f}")

    stable_democracies = ["Denmark", "Sweden", "Norway", "Switzerland", "Finland",
                          "Germany", "Canada", "New Zealand", "Uruguay", "Belgium",
                          "Iceland", "Australia", "Ireland", "Netherlands"]
    stable_fp = fp[fp["country_name"].isin(stable_democracies)]
    print(f"\n  False alerts in stable democracies: {len(stable_fp)}")
    if len(stable_fp) > 0:
        for _, r in stable_fp.iterrows():
            print(f"    {r['country_name']} ({int(r['year'])}): CSD={r['csd_index']:.2f}")

    print(f"\n{'='*60}")
    print(f"Out-of-sample alerts (post-{TRAIN_CUTOFF})")
    print(f"{'='*60}\n")

    oos = ews_df[(ews_df["year"] > TRAIN_CUTOFF) & (ews_df["ews_alert"])].sort_values(["year", "country_name"])
    for year in sorted(oos["year"].unique()):
        yr_alerts = oos[oos["year"] == year].sort_values("csd_index", ascending=False)
        entries = []
        for _, r in yr_alerts.iterrows():
            entries.append(f"{r['country_name']}({r['csd_index']:.1f})")
        print(f"  {int(year)} ({len(yr_alerts)}): {', '.join(entries)}")

    print(f"\n{'='*60}")
    print(f"CSD index rankings (2025)")
    print(f"{'='*60}\n")

    latest = ews_df[ews_df["year"] == ews_df["year"].max()].sort_values("csd_index", ascending=False)
    print("Top 15 (highest democratic stress):")
    for _, r in latest.head(15).iterrows():
        alert = " ***" if r["ews_alert"] else ""
        print(f"  {r['country_name']}: CSD={r['csd_index']:.2f}, "
              f"var_z={r['var_z']:.2f}, ar1_z={r['ar1_z']:.2f}{alert}")

    print(f"\nBottom 10 (most stable):")
    for _, r in latest.tail(10).iterrows():
        print(f"  {r['country_name']}: CSD={r['csd_index']:.2f}")

    print(f"\n{'='*60}")
    print(f"Case studies")
    print(f"{'='*60}\n")

    for country in ["Hungary", "Türkiye", "Poland", "United States of America", "Denmark"]:
        sub = ews_df[ews_df["country_name"] == country].sort_values("year").tail(10)
        if len(sub) == 0:
            continue
        print(f"{country}:")
        for _, r in sub.iterrows():
            alert = " *** ALERT" if r["ews_alert"] else ""
            oos_tag = " (OOS)" if r["year"] > TRAIN_CUTOFF else ""
            print(f"  {int(r['year'])}{oos_tag}: CSD={r['csd_index']:.2f}, "
                  f"var_z={r['var_z']:.2f}, ar1_z={r['ar1_z']:.2f}, "
                  f"Δvar={r['var_trend']:.2f}, ΔAR1={r['ar1_trend']:.2f}, "
                  f"factors={int(r['n_factor_alerts'])}/4{alert}")
        print()

    return ews_df


if __name__ == "__main__":
    ews_df = run_ews()
