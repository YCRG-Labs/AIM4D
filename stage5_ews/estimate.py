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
Z_CAP = 10.0
MIN_ABS_VAR_PCTL = 0.30
PERSISTENCE = 2
LEAD_YEARS = 5

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


def rolling_stats(series, window=WINDOW, min_w=MIN_WINDOW):
    n = len(series)
    r_var = np.full(n, np.nan)
    r_ar1 = np.full(n, np.nan)
    r_kurt = np.full(n, np.nan)

    for t in range(min_w, n):
        start = max(0, t - window)
        c = series[start:t + 1]
        if len(c) < min_w:
            continue
        r_var[t] = np.var(c, ddof=1) if len(c) > 1 else 0
        if len(c) >= 3 and np.std(c) > 1e-10:
            r_ar1[t] = np.corrcoef(c[:-1], c[1:])[0, 1]
        if len(c) >= 4:
            m = np.mean(c)
            s = np.std(c, ddof=1)
            if s > 1e-10:
                r_kurt[t] = np.mean(((c - m) / s) ** 4) - 3

    return r_var, r_ar1, r_kurt


def country_z(values, years, baseline_end=BASELINE_END):
    base = np.array(years) <= baseline_end
    vb = values[base & ~np.isnan(values)]
    if len(vb) < 3:
        va = values[~np.isnan(values)]
        if len(va) < 3:
            return np.full(len(values), 0.0)
        mu, sig = np.mean(va), np.std(va)
    else:
        mu, sig = np.mean(vb), np.std(vb)
    if sig < 1e-10:
        sig = np.std(values[~np.isnan(values)])
    if sig < 1e-10:
        return np.full(len(values), 0.0)
    return np.clip((values - mu) / sig, -Z_CAP, Z_CAP)


def rolling_kendall(values, window=8):
    n = len(values)
    taus = np.full(n, np.nan)
    for t in range(5, n):
        start = max(0, t - window)
        c = values[start:t + 1]
        v = ~np.isnan(c)
        if v.sum() >= 4:
            tau, _ = stats.kendalltau(np.arange(v.sum()), c[v])
            taus[t] = tau
    return taus


def persistence_filter(alerts, min_c=PERSISTENCE):
    out = np.zeros(len(alerts), dtype=bool)
    cnt = 0
    for i in range(len(alerts)):
        if alerts[i]:
            cnt += 1
            if cnt >= min_c:
                out[i] = True
                if cnt == min_c:
                    for j in range(max(0, i - min_c + 1), i):
                        out[j] = True
        else:
            cnt = 0
    return out


def run_ews():
    print("=== Stage 5: Early Warning Signals ===\n")

    df = load_residuals()
    countries = sorted(df["country_name"].unique())
    resid_cols = [f"resid_{k+1}" for k in range(4)]
    print(f"Countries: {len(countries)}")
    print(f"Indicators: rolling variance, AR(1), kurtosis")
    print(f"Country-relative z-scores (baseline ≤{BASELINE_END}), z-cap={Z_CAP}")

    all_train_vars = []
    for country in countries:
        cdf = df[df["country_name"] == country].sort_values("year")
        if len(cdf) < MIN_WINDOW + 2:
            continue
        for rc in resid_cols:
            rv, _, _ = rolling_stats(cdf[rc].values)
            tv = rv[np.array(cdf["year"].values) <= TRAIN_CUTOFF]
            all_train_vars.extend(tv[~np.isnan(tv)])
    abs_var_floor = np.percentile(all_train_vars, MIN_ABS_VAR_PCTL * 100)

    all_ews = []
    for ci, country in enumerate(countries):
        cdf = df[df["country_name"] == country].sort_values("year")
        if len(cdf) < MIN_WINDOW + 2:
            continue

        years = cdf["year"].values
        cid = cdf["country_text_id"].iloc[0]

        factor_alerts = np.zeros(len(years), dtype=int)
        best = {m: np.full(len(years), np.nan) for m in ["var_z", "ar1_z", "kurt_z", "var_tau", "ar1_tau"]}
        max_abs = np.full(len(years), np.nan)

        for rc in resid_cols:
            rv, ra, rk = rolling_stats(cdf[rc].values)
            vz = country_z(rv, years)
            az = country_z(ra, years)
            kz = country_z(rk, years)
            vt = rolling_kendall(rv)
            at = rolling_kendall(ra)

            for t in range(len(years)):
                above_floor = not np.isnan(rv[t]) and rv[t] > abs_var_floor

                if above_floor:
                    csd = (
                        (vz[t] > Z_THRESHOLD and az[t] > Z_THRESHOLD) or
                        (vz[t] > Z_THRESHOLD and not np.isnan(at[t]) and at[t] > 0.3) or
                        (az[t] > Z_THRESHOLD and not np.isnan(vt[t]) and vt[t] > 0.3) or
                        (vz[t] > Z_THRESHOLD and kz[t] > Z_THRESHOLD)
                    )
                    if csd:
                        factor_alerts[t] += 1

                for m, v in [("var_z", vz[t]), ("ar1_z", az[t]), ("kurt_z", kz[t]),
                             ("var_tau", vt[t] if t < len(vt) else np.nan),
                             ("ar1_tau", at[t] if t < len(at) else np.nan)]:
                    if np.isnan(best[m][t]) or (not np.isnan(v) and v > best[m][t]):
                        best[m][t] = v
                if np.isnan(max_abs[t]) or (not np.isnan(rv[t]) and rv[t] > max_abs[t]):
                    max_abs[t] = rv[t]

        csd_idx = np.zeros(len(years))
        for t in range(len(years)):
            if not np.isnan(max_abs[t]) and max_abs[t] > abs_var_floor:
                c = []
                for m in ["var_z", "ar1_z", "kurt_z"]:
                    if not np.isnan(best[m][t]):
                        c.append(min(Z_CAP, max(0, best[m][t])))
                for m in ["var_tau", "ar1_tau"]:
                    if not np.isnan(best[m][t]):
                        c.append(max(0, best[m][t]) * 2)
                csd_idx[t] = np.mean(c) if c else 0

        raw = (factor_alerts >= 2) | ((factor_alerts >= 1) & (csd_idx > 3.0))
        persistent = persistence_filter(raw)

        for t in range(len(years)):
            all_ews.append({
                "country_name": country, "country_text_id": cid,
                "year": int(years[t]),
                "var_z": best["var_z"][t], "ar1_z": best["ar1_z"][t], "kurt_z": best["kurt_z"][t],
                "var_trend": best["var_tau"][t], "ar1_trend": best["ar1_tau"][t],
                "n_factors": factor_alerts[t], "csd_index": csd_idx[t],
                "raw_alert": raw[t], "ews_alert": persistent[t],
            })

    ews_df = pd.DataFrame(all_ews)
    output_dir = os.path.dirname(os.path.abspath(__file__))
    ews_df.to_csv(os.path.join(output_dir, "ews_signals.csv"), index=False)

    print(f"\n{'='*60}")
    print(f"Validation")
    print(f"{'='*60}\n")

    hits = 0
    total = 0
    for country, info in KNOWN_EPISODES.items():
        onset = info["onset"]
        pre = ews_df[(ews_df["country_name"] == country) &
                     (ews_df["year"] >= onset - LEAD_YEARS) & (ews_df["year"] < onset)]
        if len(pre) == 0:
            print(f"  {country} ({onset}): NO DATA")
            continue
        total += 1
        if pre["ews_alert"].any():
            hits += 1
            yrs = sorted(pre[pre["ews_alert"]]["year"].tolist())
            print(f"  {country} ({info['type']} {onset}): DETECTED persistent ({onset-yrs[0]}yr lead)")
        elif pre["raw_alert"].any():
            hits += 1
            yrs = sorted(pre[pre["raw_alert"]]["year"].tolist())
            print(f"  {country} ({info['type']} {onset}): DETECTED raw ({onset-yrs[0]}yr lead)")
        else:
            print(f"  {country} ({info['type']} {onset}): MISSED (csd_max={pre['csd_index'].max():.2f})")

    sens = hits / total if total > 0 else 0
    print(f"\n  Sensitivity: {hits}/{total} ({sens:.0%})")

    known_w = {}
    for c, info in KNOWN_EPISODES.items():
        for y in range(info["onset"] - LEAD_YEARS, info["peak"] + 1):
            known_w[(c, y)] = True

    alerts = ews_df[ews_df["ews_alert"]]
    tp = alerts[alerts.apply(lambda r: (r["country_name"], r["year"]) in known_w, axis=1)]
    fp = alerts[~alerts.index.isin(tp.index)]
    prec = len(tp) / len(alerts) if len(alerts) > 0 else 0
    fp_rate = len(fp) / len(ews_df)

    print(f"\n  Alerts: {len(alerts)} / {len(ews_df)} ({len(alerts)/len(ews_df):.1%})")
    print(f"  TP: {len(tp)}, FP: {len(fp)} ({fp_rate:.1%})")
    print(f"  Precision: {prec:.1%}")
    if sens > 0 and prec > 0:
        print(f"  F1: {2*prec*sens/(prec+sens):.3f}")

    stable = ["Denmark", "Sweden", "Norway", "Switzerland", "Finland",
              "Germany", "Canada", "New Zealand", "Uruguay", "Belgium",
              "Iceland", "Australia", "Ireland", "Netherlands"]
    sfp = fp[fp["country_name"].isin(stable)]
    print(f"  Stable democracy FPs: {len(sfp)}")

    print(f"\n{'='*60}")
    print(f"Case studies")
    print(f"{'='*60}\n")

    for country in ["Hungary", "Türkiye", "Poland", "United States of America", "Denmark"]:
        sub = ews_df[ews_df["country_name"] == country].sort_values("year").tail(10)
        if len(sub) == 0:
            continue
        print(f"{country}:")
        for _, r in sub.iterrows():
            a = " ***" if r["ews_alert"] else ""
            o = "(OOS) " if r["year"] > TRAIN_CUTOFF else ""
            print(f"  {int(r['year'])} {o}CSD={r['csd_index']:.1f} var_z={r['var_z']:.1f} "
                  f"ar1_z={r['ar1_z']:.1f} kurt_z={r['kurt_z']:.1f} f={int(r['n_factors'])}/4{a}")
        print()

    return ews_df


if __name__ == "__main__":
    ews_df = run_ews()
