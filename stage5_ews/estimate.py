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
    resid_path = os.path.join(base, "..", "stage4_nscm", "nscm_residuals.csv")
    scores_path = os.path.join(base, "..", "stage4_nscm", "contagion_scores.csv")
    factors_path = os.path.join(base, "..", "stage1_factors", "country_year_factors.csv")

    scores = pd.read_csv(scores_path)
    factors = pd.read_csv(factors_path)
    merged = factors.merge(
        scores[["country_text_id", "year", "contagion_score", "domestic_score"]],
        on=["country_text_id", "year"], how="inner"
    )

    factor_cols = ["factor_1", "factor_2", "factor_3", "factor_4"]
    for k, fc in enumerate(factor_cols):
        merged[f"resid_factor_{k+1}"] = merged.groupby("country_text_id")[fc].diff() * merged["domestic_score"]

    if os.path.exists(resid_path):
        resid = pd.read_csv(resid_path)
        merged = merged.merge(resid, on=["country_text_id", "year"], how="left")
        dom_cols = [c for c in resid.columns if c.startswith("nscm_resid_domestic_")]
        for k, dc in enumerate(dom_cols):
            merged[f"resid_nscm_{k+1}"] = merged[dc]
        n_nscm = len(dom_cols)
        print(f"Loaded dual residuals: {len(factor_cols)} factor-based + {n_nscm} NSCM")
    else:
        n_nscm = 0
        print(f"Factor-based residuals only ({len(factor_cols)} dimensions)")

    all_resid_cols = [f"resid_factor_{k+1}" for k in range(4)]
    if n_nscm > 0:
        all_resid_cols += [f"resid_nscm_{k+1}" for k in range(n_nscm)]

    merged = merged.dropna(subset=[f"resid_factor_{k+1}" for k in range(4)])
    return merged, all_resid_cols


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


def compute_election_vulnerability():
    base = os.path.dirname(os.path.abspath(__file__))
    vdem_path = os.path.join(base, "..", "data", "vdem_v16.csv")

    cols = ["country_name", "country_text_id", "year",
            "v2xpas_democracy_opposition", "v2xpas_exclusion_opposition",
            "v2xpas_democracy_government",
            "v2eltype_0", "v2eltype_6", "v2eltype_7",
            "v2psoppaut", "v2x_polyarchy"]

    available = pd.read_csv(vdem_path, low_memory=False, nrows=1).columns
    cols = [c for c in cols if c in available]
    vdem = pd.read_csv(vdem_path, low_memory=False, usecols=cols)
    vdem = vdem[vdem["year"] >= 1990]

    has_election = np.zeros(len(vdem))
    for et in ["v2eltype_0", "v2eltype_6", "v2eltype_7"]:
        if et in vdem.columns:
            has_election = np.maximum(has_election, vdem[et].fillna(0).values)
    vdem["has_election"] = has_election

    vdem["election_within_2yr"] = vdem.groupby("country_text_id")["has_election"].transform(
        lambda x: x.rolling(3, min_periods=1, center=True).max()
    ).fillna(0)

    if "v2xpas_democracy_opposition" in vdem.columns:
        vdem["opp_antidem"] = vdem.groupby("country_text_id")["v2xpas_democracy_opposition"].transform(
            lambda x: x.interpolate(limit_direction="both")
        )
    else:
        vdem["opp_antidem"] = np.nan

    if "v2xpas_exclusion_opposition" in vdem.columns:
        vdem["opp_exclusion"] = vdem.groupby("country_text_id")["v2xpas_exclusion_opposition"].transform(
            lambda x: x.interpolate(limit_direction="both")
        )
    else:
        vdem["opp_exclusion"] = np.nan

    if "v2psoppaut" in vdem.columns:
        vdem["opp_autonomy"] = vdem.groupby("country_text_id")["v2psoppaut"].transform(
            lambda x: x.interpolate(limit_direction="both")
        )
    else:
        vdem["opp_autonomy"] = np.nan

    vdem["party_threat"] = 0.0
    mask = ~vdem["opp_antidem"].isna()
    if mask.any():
        vdem.loc[mask, "party_threat"] += (1 - vdem.loc[mask, "opp_antidem"]).clip(0, 1) * 3
    mask = ~vdem["opp_exclusion"].isna()
    if mask.any():
        vdem.loc[mask, "party_threat"] += vdem.loc[mask, "opp_exclusion"].clip(0, 1) * 3
    mask = ~vdem["opp_autonomy"].isna()
    if mask.any():
        low_autonomy = (4 - vdem.loc[mask, "opp_autonomy"].clip(0, 4)) / 4
        vdem.loc[mask, "party_threat"] += low_autonomy * 2

    vdem["election_vulnerability"] = vdem["party_threat"] * vdem["election_within_2yr"]

    out = vdem[["country_name", "country_text_id", "year",
                "election_within_2yr", "party_threat", "election_vulnerability"]].copy()

    return out


def run_ews():
    print("=== Stage 5: Early Warning Signals ===\n")

    df, resid_cols = load_residuals()
    countries = sorted(df["country_name"].unique())
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

        raw = (factor_alerts >= 3) | ((factor_alerts >= 2) & (csd_idx > 2.5)) | ((factor_alerts >= 1) & (csd_idx > 4.0))
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
    print(f"Election vulnerability module")
    print(f"{'='*60}\n")

    elec_vuln = compute_election_vulnerability()
    ews_df = ews_df.merge(elec_vuln[["country_text_id", "year", "election_within_2yr",
                                      "party_threat", "election_vulnerability"]],
                           on=["country_text_id", "year"], how="left")
    ews_df["election_vulnerability"] = ews_df["election_vulnerability"].fillna(0)
    ews_df["party_threat"] = ews_df["party_threat"].fillna(0)
    ews_df["election_within_2yr"] = ews_df["election_within_2yr"].fillna(0)

    ews_df["combined_risk"] = ews_df["csd_index"] + ews_df["election_vulnerability"] * 0.5
    ev_train = ews_df[ews_df["year"] <= TRAIN_CUTOFF]["election_vulnerability"]
    ev_high = ev_train.quantile(0.80)
    ev_moderate = ev_train[ev_train > 0].quantile(0.50)
    print(f"  Election thresholds: high={ev_high:.2f} (p80), moderate={ev_moderate:.2f} (p50 of nonzero)")
    ews_df["election_alert"] = (
        (ews_df["election_vulnerability"] > ev_high) |
        ((ews_df["party_threat"] > 4.5) & (ews_df["election_within_2yr"] > 0) & (ews_df["csd_index"] > 1.0))
    )
    ews_df["combined_alert"] = ews_df["ews_alert"] | ews_df["election_alert"]

    n_elec_alerts = ews_df["election_alert"].sum()
    print(f"  Election alerts: {n_elec_alerts}")

    for country in ["Poland", "Hungary", "Venezuela", "Türkiye"]:
        sub = ews_df[ews_df["country_name"] == country].sort_values("year")
        elec_hits = sub[sub["election_alert"]]
        if len(elec_hits) > 0:
            yrs = sorted(elec_hits["year"].tolist())
            print(f"  {country}: election alerts in {yrs[:5]}")
        else:
            peak_ev = sub["election_vulnerability"].max()
            print(f"  {country}: no election alert (max vuln={peak_ev:.2f})")

    print(f"\n{'='*60}")
    print(f"Validation (CSD + election combined)")
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
        if pre["combined_alert"].any():
            hits += 1
            yrs = sorted(pre[pre["combined_alert"]]["year"].tolist())
            source = "CSD" if pre["ews_alert"].any() else "election"
            if pre["ews_alert"].any() and pre["election_alert"].any():
                source = "CSD+election"
            print(f"  {country} ({info['type']} {onset}): DETECTED via {source} ({onset-yrs[0]}yr lead)")
        else:
            print(f"  {country} ({info['type']} {onset}): MISSED (csd={pre['csd_index'].max():.2f}, "
                  f"elec_vuln={pre['election_vulnerability'].max():.2f})")

    sens = hits / total if total > 0 else 0
    print(f"\n  Sensitivity: {hits}/{total} ({sens:.0%})")

    known_w = {}
    for c, info in KNOWN_EPISODES.items():
        for y in range(info["onset"] - LEAD_YEARS, info["onset"] + 1):
            known_w[(c, y)] = True

    alerts = ews_df[ews_df["combined_alert"]]
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
    print(f"Continuous risk score evaluation (AUC)")
    print(f"{'='*60}\n")

    from sklearn.metrics import roc_auc_score, average_precision_score

    ews_eval = ews_df.copy()
    ews_eval["label"] = ews_eval.apply(
        lambda r: 1 if (r["country_name"], r["year"]) in known_w else 0, axis=1
    )
    valid = ews_eval.dropna(subset=["csd_index"])
    if valid["label"].sum() > 0 and valid["label"].nunique() > 1:
        auc_roc = roc_auc_score(valid["label"], valid["combined_risk"])
        auc_pr = average_precision_score(valid["label"], valid["combined_risk"])
        base_rate = valid["label"].mean()
        lift = auc_pr / base_rate

        print(f"  Base rate (positive class): {base_rate:.3f}")
        print(f"  AUC-ROC: {auc_roc:.3f}")
        print(f"  AUC-PR:  {auc_pr:.3f}")
        print(f"  Lift over random: {lift:.1f}x")

        oos = valid[valid["year"] > TRAIN_CUTOFF]
        if oos["label"].sum() > 0 and oos["label"].nunique() > 1:
            auc_roc_oos = roc_auc_score(oos["label"], oos["combined_risk"])
            auc_pr_oos = average_precision_score(oos["label"], oos["combined_risk"])
            print(f"  AUC-ROC (OOS): {auc_roc_oos:.3f}")
            print(f"  AUC-PR (OOS):  {auc_pr_oos:.3f}")

        top_pctiles = [99, 95, 90, 80]
        print(f"\n  Risk score calibration:")
        for p in top_pctiles:
            thresh = valid["combined_risk"].quantile(p / 100)
            flagged = valid[valid["combined_risk"] >= thresh]
            if len(flagged) > 0:
                prec_at_p = flagged["label"].mean()
                recall_at_p = flagged["label"].sum() / valid["label"].sum()
                print(f"    Top {100-p}%: precision={prec_at_p:.1%}, recall={recall_at_p:.1%} (thresh={thresh:.2f})")

    print(f"\n{'='*60}")
    print(f"Case studies")
    print(f"{'='*60}\n")

    for country in ["Hungary", "Türkiye", "Poland", "United States of America", "Denmark"]:
        sub = ews_df[ews_df["country_name"] == country].sort_values("year").tail(10)
        if len(sub) == 0:
            continue
        print(f"{country}:")
        for _, r in sub.iterrows():
            flags = []
            if r.get("ews_alert", False):
                flags.append("CSD")
            if r.get("election_alert", False):
                flags.append("ELEC")
            a = f" *** {'+'.join(flags)}" if flags else ""
            o = "(OOS) " if r["year"] > TRAIN_CUTOFF else ""
            ev = r.get("election_vulnerability", 0)
            print(f"  {int(r['year'])} {o}CSD={r['csd_index']:.1f} elec_vuln={ev:.1f} "
                  f"var_z={r['var_z']:.1f} ar1_z={r['ar1_z']:.1f} f={int(r['n_factors'])}/9{a}")
        print()

    return ews_df


if __name__ == "__main__":
    ews_df = run_ews()
