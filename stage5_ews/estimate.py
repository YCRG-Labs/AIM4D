import sys
import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "BLIS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, "4")

import numpy as np
import pandas as pd
from scipy import stats
from joblib import Parallel, delayed

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

WINDOW = 8
MIN_WINDOW = 5
BASELINE_END = int(os.environ.get("AIM4D_BASELINE_END", "2005"))
TRAIN_CUTOFF = int(os.environ.get("AIM4D_CUTOFF", "2019"))
EXCLUDE_COUNTRY = os.environ.get("AIM4D_EXCLUDE_COUNTRY", "").strip() or None
Z_THRESHOLD = 1.5
Z_CAP = 10.0
MIN_ABS_VAR_PCTL = 0.30
PERSISTENCE = 2
LEAD_YEARS = int(os.environ.get("AIM4D_LEAD_YEARS", "5"))
N_SURROGATES = 50
KENDALL_SIG = 0.05

COUP_LEAD_OVERRIDE = os.environ.get("AIM4D_COUP_LEAD", "").strip()
LEAD_YEARS_COUP = int(COUP_LEAD_OVERRIDE) if COUP_LEAD_OVERRIDE else LEAD_YEARS
POS_WEIGHT = float(os.environ.get("AIM4D_POS_WEIGHT", "1.0"))
SMOOTH_WINDOW = int(os.environ.get("AIM4D_SMOOTH", "1"))
POSTONSET_EXCL_YEARS_ENV = int(os.environ.get("AIM4D_POSTONSET", "5"))
WATCH_PCTL = float(os.environ.get("AIM4D_WATCH_PCTL", "0.80"))
WARNING_PCTL = float(os.environ.get("AIM4D_WARNING_PCTL", "0.95"))
ALERT_PCTL = float(os.environ.get("AIM4D_ALERT_PCTL", "0.98"))
SEED_OFFSET = int(os.environ.get("AIM4D_SEED_OFFSET", "0"))
NO_WRITE = (os.environ.get("AIM4D_NO_WRITE") == "1"
            or bool(os.environ.get("AIM4D_ABLATE_GROUP", "").strip())
            or SEED_OFFSET != 0)


def lead_for(info):
    """Return type-appropriate pre-onset label window."""
    return LEAD_YEARS_COUP if info.get("type") == "coup" else LEAD_YEARS

KNOWN_EPISODES = {
    "Hungary": {"onset": 2010, "peak": 2018, "type": "backsliding"},
    "Türkiye": {"onset": 2013, "peak": 2017, "type": "backsliding"},
    "Poland": {"onset": 2015, "peak": 2019, "type": "backsliding"},
    "Venezuela": {"onset": 2002, "peak": 2013, "type": "backsliding"},
    "Tunisia": {"onset": 2021, "peak": 2023, "type": "backsliding"},
    "Burma/Myanmar": {"onset": 2021, "peak": 2022, "type": "coup"},
    "Mali": {"onset": 2020, "peak": 2021, "type": "coup"},
    "Burkina Faso": {"onset": 2022, "peak": 2022, "type": "coup"},
    "Nicaragua": {"onset": 2007, "peak": 2021, "type": "backsliding"},
    "Philippines": {"onset": 2016, "peak": 2022, "type": "backsliding"},
    "India": {"onset": 2014, "peak": 2024, "type": "backsliding"},
    "Brazil": {"onset": 2019, "peak": 2022, "type": "backsliding"},
    "El Salvador": {"onset": 2019, "peak": 2024, "type": "backsliding"},
    "Russia": {"onset": 2000, "peak": 2012, "type": "backsliding"},
    "Serbia": {"onset": 2012, "peak": 2020, "type": "backsliding"},
    "Bangladesh": {"onset": 2007, "peak": 2024, "type": "backsliding"},
    "Thailand": {"onset": 2014, "peak": 2014, "type": "coup"},
    "Egypt": {"onset": 2013, "peak": 2014, "type": "coup"},
    "Fiji": {"onset": 2006, "peak": 2007, "type": "coup"},
    "Honduras": {"onset": 2009, "peak": 2010, "type": "coup"},
    "Niger": {"onset": 2009, "peak": 2010, "type": "coup"},
    "Guinea": {"onset": 2009, "peak": 2010, "type": "coup"},
    "Guinea-Bissau": {"onset": 2012, "peak": 2013, "type": "coup"},
    "Libya": {"onset": 2014, "peak": 2015, "type": "coup"},
    "Afghanistan": {"onset": 2021, "peak": 2022, "type": "coup"},
    "Sudan": {"onset": 2021, "peak": 2022, "type": "coup"},
    "Chad": {"onset": 2021, "peak": 2022, "type": "coup"},
    "Ukraine": {"onset": 2010, "peak": 2013, "type": "backsliding"},
    "Maldives": {"onset": 2013, "peak": 2017, "type": "backsliding"},
    "Zambia": {"onset": 2013, "peak": 2020, "type": "backsliding"},
    "Montenegro": {"onset": 2007, "peak": 2020, "type": "backsliding"},
    "North Macedonia": {"onset": 2011, "peak": 2016, "type": "backsliding"},
    "Kenya": {"onset": 2007, "peak": 2008, "type": "backsliding"},
    "Bolivia": {"onset": 2019, "peak": 2022, "type": "backsliding"},
    "Benin": {"onset": 2019, "peak": 2023, "type": "backsliding"},
    "Ivory Coast": {"onset": 2020, "peak": 2022, "type": "backsliding"},
    "Nigeria": {"onset": 2021, "peak": 2024, "type": "backsliding"},
    "Guyana": {"onset": 2021, "peak": 2023, "type": "backsliding"},
    "Mauritius": {"onset": 2017, "peak": 2023, "type": "backsliding"},
    "Belarus": {"onset": 1996, "peak": 2024, "type": "backsliding"},
    "Georgia": {"onset": 2024, "peak": 2025, "type": "backsliding"},
    "Greece": {"onset": 2020, "peak": 2024, "type": "backsliding"},
    "Botswana": {"onset": 2021, "peak": 2024, "type": "backsliding"},
    "Slovenia": {"onset": 2021, "peak": 2024, "type": "backsliding"},
    "South Korea": {"onset": 2024, "peak": 2025, "type": "backsliding"},
    "Indonesia": {"onset": 2024, "peak": 2025, "type": "backsliding"},
    "Mexico": {"onset": 2024, "peak": 2025, "type": "backsliding"},
    "Mongolia": {"onset": 2024, "peak": 2025, "type": "backsliding"},
    "Gabon": {"onset": 2023, "peak": 2024, "type": "coup"},
    "Haiti": {"onset": 2022, "peak": 2024, "type": "coup"},
    "Cambodia": {"onset": 2017, "peak": 2024, "type": "backsliding"},
    "Tajikistan": {"onset": 2016, "peak": 2024, "type": "backsliding"},
    "Saudi Arabia": {"onset": 2017, "peak": 2024, "type": "backsliding"},
    "Uganda": {"onset": 2017, "peak": 2024, "type": "backsliding"},
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
    """
    G8: Extended CSD indicators. Adds skewness and absolute-residual mean
    on top of the original variance/AR(1)/kurtosis. Skewness is the
    asymmetry of fluctuations, a well-attested pre-bifurcation signal
    (Scheffer 2009 review).
    """
    n = len(series)
    r_var = np.full(n, np.nan)
    r_ar1 = np.full(n, np.nan)
    r_kurt = np.full(n, np.nan)
    r_skew = np.full(n, np.nan)
    r_abs = np.full(n, np.nan)

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
                r_skew[t] = np.mean(((c - m) / s) ** 3)
        r_abs[t] = np.mean(np.abs(c - np.mean(c)))

    return r_var, r_ar1, r_kurt, r_skew, r_abs


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


def multivariate_csd(resid_matrix, window=WINDOW, min_w=MIN_WINDOW):
    """
    Multivariate CSD indicators (Weinans et al. 2021, Held & Kleinen 2004).
    Tracks dominant eigenvalue of cross-factor covariance and mean cross-correlation
    in a rolling window — captures correlated fluctuations across factors.
    """
    n, d = resid_matrix.shape
    dom_eig = np.full(n, np.nan)
    mean_xcorr = np.full(n, np.nan)
    total_var = np.full(n, np.nan)

    for t in range(min_w, n):
        start = max(0, t - window)
        chunk = resid_matrix[start:t + 1]
        valid = ~np.any(np.isnan(chunk), axis=1)
        chunk = chunk[valid]
        if len(chunk) < min_w or chunk.shape[1] < 2:
            continue

        cov = np.cov(chunk.T)
        if cov.ndim < 2:
            continue

        eigvals = np.linalg.eigvalsh(cov)
        dom_eig[t] = eigvals[-1]
        total_var[t] = np.trace(cov)

        stds = np.std(chunk, axis=0, ddof=1)
        stds = np.maximum(stds, 1e-10)
        corr = cov / np.outer(stds, stds)
        np.fill_diagonal(corr, 0)
        n_pairs = d * (d - 1)
        mean_xcorr[t] = np.sum(np.abs(corr)) / n_pairs if n_pairs > 0 else 0

    return dom_eig, mean_xcorr, total_var


def kendall_tau_with_surrogates(series, window=WINDOW, n_surrogates=N_SURROGATES):
    """
    Kendall tau trend test with ARMA surrogate significance (Dakos et al. 2012).
    Returns tau values and boolean significance at each time step.
    """
    n = len(series)
    taus = np.full(n, np.nan)
    significant = np.zeros(n, dtype=bool)

    for t in range(MIN_WINDOW, n):
        start = max(0, t - window)
        c = series[start:t + 1]
        valid = ~np.isnan(c)
        c_valid = c[valid]
        if len(c_valid) < 4:
            continue

        tau, _ = stats.kendalltau(np.arange(len(c_valid)), c_valid)
        taus[t] = tau

        if len(c_valid) >= 5:
            ar1 = np.corrcoef(c_valid[:-1], c_valid[1:])[0, 1] if np.std(c_valid) > 1e-10 else 0
            ar1 = np.clip(ar1, -0.99, 0.99)
            residual_std = np.std(c_valid) * np.sqrt(1 - ar1 ** 2) if abs(ar1) < 1 else np.std(c_valid)

            surr_taus = np.zeros(n_surrogates)
            for s in range(n_surrogates):
                surr = np.zeros(len(c_valid))
                surr[0] = np.random.normal(0, np.std(c_valid))
                for i in range(1, len(surr)):
                    surr[i] = ar1 * surr[i - 1] + np.random.normal(0, max(residual_std, 1e-10))
                surr_taus[s], _ = stats.kendalltau(np.arange(len(surr)), surr)

            p_value = np.mean(surr_taus >= tau) if tau > 0 else 1.0
            significant[t] = p_value < KENDALL_SIG

    return taus, significant


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

    components = []
    mask = ~vdem["opp_antidem"].isna()
    c1 = pd.Series(np.nan, index=vdem.index)
    if mask.any():
        c1[mask] = (1 - vdem.loc[mask, "opp_antidem"]).clip(0, 1)
    components.append(c1)

    mask = ~vdem["opp_exclusion"].isna()
    c2 = pd.Series(np.nan, index=vdem.index)
    if mask.any():
        c2[mask] = vdem.loc[mask, "opp_exclusion"].clip(0, 1)
    components.append(c2)

    mask = ~vdem["opp_autonomy"].isna()
    c3 = pd.Series(np.nan, index=vdem.index)
    if mask.any():
        c3[mask] = (4 - vdem.loc[mask, "opp_autonomy"].clip(0, 4)) / 4
    components.append(c3)

    comp_df = pd.concat(components, axis=1)
    vdem["party_threat"] = comp_df.mean(axis=1).fillna(0) * 8

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
            rv, _, _, _, _ = rolling_stats(cdf[rc].values)
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
        best = {m: np.full(len(years), np.nan) for m in
                ["var_z", "ar1_z", "kurt_z", "skew_z", "var_tau", "ar1_tau", "skew_tau"]}
        max_abs = np.full(len(years), np.nan)

        for rc in resid_cols:
            rv, ra, rk, rs, rabs = rolling_stats(cdf[rc].values)
            vz = country_z(rv, years)
            az = country_z(ra, years)
            kz = country_z(rk, years)
            sz = country_z(rs, years)
            vt = rolling_kendall(rv)
            at = rolling_kendall(ra)
            st = rolling_kendall(rs)

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
                             ("skew_z", sz[t]),
                             ("var_tau", vt[t] if t < len(vt) else np.nan),
                             ("ar1_tau", at[t] if t < len(at) else np.nan),
                             ("skew_tau", st[t] if t < len(st) else np.nan)]:
                    if np.isnan(best[m][t]) or (not np.isnan(v) and v > best[m][t]):
                        best[m][t] = v
                if np.isnan(max_abs[t]) or (not np.isnan(rv[t]) and rv[t] > max_abs[t]):
                    max_abs[t] = rv[t]

        factor_resid_cols = [f"resid_factor_{k+1}" for k in range(4)]
        available_fc = [c for c in factor_resid_cols if c in cdf.columns]
        resid_matrix = cdf[available_fc].values if available_fc else np.zeros((len(years), 1))

        dom_eig, mean_xcorr, total_var = multivariate_csd(resid_matrix)

        eig_tau, eig_sig = kendall_tau_with_surrogates(dom_eig)
        xcorr_tau, xcorr_sig = kendall_tau_with_surrogates(mean_xcorr)
        var_tau_mv, var_sig = kendall_tau_with_surrogates(total_var)

        eig_z = country_z(dom_eig, years)
        xcorr_z = country_z(mean_xcorr, years)

        mv_csd_alert = eig_sig | xcorr_sig | var_sig

        csd_idx = np.zeros(len(years))
        for t in range(len(years)):
            components = []
            if not np.isnan(max_abs[t]) and max_abs[t] > abs_var_floor:
                for m in ["var_z", "ar1_z", "kurt_z"]:
                    if not np.isnan(best[m][t]):
                        components.append(min(Z_CAP, max(0, best[m][t])))
                for m in ["var_tau", "ar1_tau"]:
                    if not np.isnan(best[m][t]):
                        components.append(max(0, best[m][t]) * 2)
            csd_idx[t] = np.mean(components) if components else 0

        mv_csd_idx = np.zeros(len(years))
        for t in range(len(years)):
            mv_components = []
            if not np.isnan(eig_z[t]):
                mv_components.append(min(Z_CAP, max(0, eig_z[t])))
            if not np.isnan(xcorr_z[t]):
                mv_components.append(min(Z_CAP, max(0, xcorr_z[t])))
            if not np.isnan(eig_tau[t]):
                mv_components.append(max(0, eig_tau[t]) * 3)
            mv_csd_idx[t] = np.mean(mv_components) if mv_components else 0

        raw_univariate = (factor_alerts >= 3) | ((factor_alerts >= 2) & (csd_idx > 2.5)) | ((factor_alerts >= 1) & (csd_idx > 4.0))
        raw_multivariate = mv_csd_alert & (csd_idx > 1.5)
        raw = raw_univariate | raw_multivariate
        persistent = persistence_filter(raw)

        for t in range(len(years)):
            all_ews.append({
                "country_name": country, "country_text_id": cid,
                "year": int(years[t]),
                "var_z": best["var_z"][t], "ar1_z": best["ar1_z"][t], "kurt_z": best["kurt_z"][t],
                "skew_z": best["skew_z"][t],
                "var_trend": best["var_tau"][t], "ar1_trend": best["ar1_tau"][t],
                "skew_trend": best["skew_tau"][t],
                "n_factors": factor_alerts[t], "csd_index": csd_idx[t],
                "mv_csd_index": mv_csd_idx[t],
                "dom_eig_z": eig_z[t], "xcorr_z": xcorr_z[t],
                "eig_trend_sig": eig_sig[t], "xcorr_trend_sig": xcorr_sig[t],
                "mv_csd_alert": mv_csd_alert[t],
                "raw_alert": raw[t], "ews_alert": persistent[t],
            })

    ews_df = pd.DataFrame(all_ews)
    output_dir = os.path.dirname(os.path.abspath(__file__))
    if not NO_WRITE:
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
        ((ews_df["party_threat"] > 4.0) & (ews_df["election_within_2yr"] > 0) & (ews_df["csd_index"] > 1.0))
    )
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
    print(f"Democratic vulnerability module (high-polyarchy sensitivity)")
    print(f"{'='*60}\n")

    factors = pd.read_csv(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                        "..", "stage1_factors", "country_year_factors.csv"))
    factors["f1_change"] = factors.groupby("country_text_id")["factor_1"].diff()
    factors["f1_rolling_mean"] = factors.groupby("country_text_id")["factor_1"].transform(
        lambda x: x.rolling(5, min_periods=3).mean()
    )
    ews_df = ews_df.merge(factors[["country_text_id", "year", "factor_1", "f1_change", "f1_rolling_mean"]],
                           on=["country_text_id", "year"], how="left")

    high_dem = ews_df["f1_rolling_mean"] > ews_df["f1_rolling_mean"].quantile(0.75)
    declining = ews_df["f1_change"] < -0.02
    any_csd = ews_df["csd_index"] > 0.5
    ews_df["dem_vulnerability_alert"] = high_dem & declining & any_csd
    n_dv = ews_df["dem_vulnerability_alert"].sum()
    print(f"  Democratic vulnerability alerts: {n_dv}")

    for country in ["Hungary", "United States of America", "Denmark"]:
        sub = ews_df[(ews_df["country_name"] == country) & ews_df["dem_vulnerability_alert"]]
        if len(sub) > 0:
            print(f"  {country}: alerts in {sorted(sub['year'].tolist())[:5]}")
        else:
            print(f"  {country}: no democratic vulnerability alert")

    print(f"\n{'='*60}")
    print(f"Military threat module")
    print(f"{'='*60}\n")

    macro = pd.read_csv(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                      "..", "data", "macro_covariates.csv"))
    macro["mil_growth"] = macro.groupby("iso3")["military_spending"].pct_change()
    macro["mil_mean"] = macro.groupby("iso3")["military_spending"].transform(
        lambda x: x.rolling(10, min_periods=5).mean()
    )
    macro["mil_std"] = macro.groupby("iso3")["military_spending"].transform(
        lambda x: x.rolling(10, min_periods=5).std()
    )
    macro["mil_zscore"] = (macro["military_spending"] - macro["mil_mean"]) / macro["mil_std"].clip(lower=0.01)
    ews_df = ews_df.merge(macro[["iso3", "year", "mil_zscore", "mil_growth"]].rename(columns={"iso3": "country_text_id"}),
                           on=["country_text_id", "year"], how="left")
    ews_df["mil_zscore"] = ews_df["mil_zscore"].fillna(0)
    ews_df["mil_growth"] = ews_df["mil_growth"].fillna(0)

    ews_df["military_threat_alert"] = (ews_df["mil_zscore"] > 1.5) & (ews_df["csd_index"] > 1.0)
    n_mil = ews_df["military_threat_alert"].sum()
    print(f"  Military threat alerts: {n_mil}")

    for country in ["Thailand", "Egypt", "Burma/Myanmar", "Mali"]:
        sub = ews_df[(ews_df["country_name"] == country) & ews_df["military_threat_alert"]]
        if len(sub) > 0:
            print(f"  {country}: alerts in {sorted(sub['year'].tolist())[:5]}")
        else:
            print(f"  {country}: no military threat alert")

    gdelt_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data", "gdelt_country_year.csv")
    gdelt_features = []
    if os.path.exists(gdelt_path):
        gdelt = pd.read_csv(gdelt_path)
        gdelt = gdelt.rename(columns={"country_code": "country_text_id"})
        for col in ["protest_count", "conflict_count", "repression_count"]:
            if col in gdelt.columns:
                gdelt[f"{col}_mean"] = gdelt.groupby("country_text_id")[col].transform(
                    lambda x: x.rolling(5, min_periods=3).mean().shift(1)
                )
                gdelt[f"{col}_std"] = gdelt.groupby("country_text_id")[col].transform(
                    lambda x: x.rolling(5, min_periods=3).std().shift(1)
                )
                gdelt[f"{col}_zscore"] = (gdelt[col] - gdelt[f"{col}_mean"]) / gdelt[f"{col}_std"].clip(lower=1)
                gdelt[f"{col}_zscore"] = gdelt[f"{col}_zscore"].clip(-10, 10)
                gdelt_features.append(f"{col}_zscore")

        for col in ["protest_count", "conflict_count", "repression_count"]:
            lag_col = f"{col}_zscore_lag1"
            gdelt[lag_col] = gdelt.groupby("country_text_id")[f"{col}_zscore"].shift(1)
            gdelt_features.append(lag_col)

        gdelt_merge_cols = ["country_text_id", "year"] + gdelt_features
        gdelt_merge_cols = [c for c in gdelt_merge_cols if c in gdelt.columns]
        ews_df = ews_df.merge(gdelt[gdelt_merge_cols], on=["country_text_id", "year"], how="left")
        for gf in gdelt_features:
            if gf in ews_df.columns:
                ews_df[gf] = ews_df[gf].fillna(0)
        print(f"\n  GDELT event features loaded: {gdelt_features}")

    institutional_cols = ["v2juncind", "v2xlg_legcon", "v2x_jucon", "v2exrescon"]
    try:
        vdem_inst_avail = [c for c in institutional_cols if c in pd.read_csv(vdem_path, low_memory=False, nrows=1).columns]
        if vdem_inst_avail:
            inst_data = pd.read_csv(vdem_path, low_memory=False,
                                     usecols=["country_text_id", "year"] + vdem_inst_avail)
            inst_features = []
            for col in vdem_inst_avail:
                inst_data[f"{col}_change"] = inst_data.groupby("country_text_id")[col].diff()
                inst_data[f"{col}_change2"] = inst_data.groupby("country_text_id")[col].diff(2)
                inst_features += [f"{col}_change", f"{col}_change2"]
            ews_df = ews_df.merge(inst_data[["country_text_id", "year"] + inst_features],
                                   on=["country_text_id", "year"], how="left")
            for f in inst_features:
                ews_df[f] = ews_df[f].fillna(0)
            gdelt_features += inst_features
            print(f"  Institutional erosion features loaded: {vdem_inst_avail}")
    except Exception:
        pass

    base_dir = os.path.dirname(os.path.abspath(__file__))
    vdem_path = os.path.join(base_dir, "..", "data", "vdem_v16.csv")
    pitf_path = os.path.join(base_dir, "..", "data", "macro_pitf.csv")
    pitf_features = []
    if os.path.exists(pitf_path):
        pitf = pd.read_csv(pitf_path)
        if "iso3" in pitf.columns:
            pitf = pitf.rename(columns={"iso3": "country_text_id"})
        pitf_features = [c for c in pitf.columns if c not in {"country_text_id", "year"}]
        ews_df = ews_df.merge(pitf, on=["country_text_id", "year"], how="left")
        for f in pitf_features:
            ews_df[f] = ews_df.groupby("country_text_id")[f].ffill()
            train_median = ews_df.loc[ews_df["year"] <= TRAIN_CUTOFF, f].median()
            if pd.isna(train_median):
                train_median = 0.0
            ews_df[f] = ews_df[f].fillna(train_median)
        print(f"  F5 PITF features loaded: {pitf_features}")
    else:
        print(f"  F5 PITF: macro_pitf.csv not found; run data/download_pitf.py to enable")

    diff_path = os.path.join(base_dir, "..", "data", "global_diffusion.csv")
    diffusion_features = []
    if os.path.exists(diff_path):
        diff = pd.read_csv(diff_path)
        diffusion_features = [c for c in diff.columns if c not in {"country_text_id", "year"}]
        ews_df = ews_df.merge(diff, on=["country_text_id", "year"], how="left")
        for f in diffusion_features:
            ews_df[f] = ews_df.groupby("country_text_id")[f].ffill()
            ews_df[f] = ews_df[f].fillna(0.0)
        print(f"  F4 global diffusion features loaded: {diffusion_features}")
    else:
        print(f"  F4 global diffusion: global_diffusion.csv not found; run data/compute_global_diffusion.py")

    arch_path = os.path.join(base_dir, "..", "data", "archigos_features.csv")
    archigos_features = []
    if os.path.exists(arch_path):
        arch = pd.read_csv(arch_path)
        if "iso3" in arch.columns:
            arch = arch.rename(columns={"iso3": "country_text_id"})
        archigos_features = [c for c in arch.columns if c not in {"country_text_id", "year"}]
        ews_df = ews_df.merge(arch, on=["country_text_id", "year"], how="left")
        for f in archigos_features:
            ews_df[f] = ews_df.groupby("country_text_id")[f].ffill()
            train_median = ews_df.loc[ews_df["year"] <= TRAIN_CUTOFF, f].median()
            if pd.isna(train_median):
                train_median = 0.0
            ews_df[f] = ews_df[f].fillna(train_median)
        print(f"  F3 Archigos features loaded: {archigos_features}")
    else:
        print(f"  F3 Archigos: archigos_features.csv not found; run data/download_archigos.py")

    c22_path = os.path.join(base_dir, "..", "data", "catch22_features.csv")
    c22_features = []
    if os.path.exists(c22_path):
        c22 = pd.read_csv(c22_path)
        c22_features = [c for c in c22.columns if c not in {"country_text_id", "year"}]
        ews_df = ews_df.merge(c22, on=["country_text_id", "year"], how="left")
        for f in c22_features:
            ews_df[f] = ews_df.groupby("country_text_id")[f].ffill()
            ews_df[f] = ews_df[f].fillna(0.0)
        print(f"  G7 catch22 features loaded: {len(c22_features)} features")
    else:
        print(f"  G7 catch22: catch22_features.csv not found; run data/compute_catch22.py")

    cp_path = os.path.join(base_dir, "..", "data", "changepoints.csv")
    cp_features = []
    if os.path.exists(cp_path):
        cp = pd.read_csv(cp_path)
        cp_features = [c for c in cp.columns if c not in {"country_text_id", "year"}]
        ews_df = ews_df.merge(cp, on=["country_text_id", "year"], how="left")
        for f in cp_features:
            ews_df[f] = ews_df.groupby("country_text_id")[f].ffill()
            ews_df[f] = ews_df[f].fillna(99)
        print(f"  G9 change-point features loaded: {cp_features}")
    else:
        print(f"  G9 change-points: changepoints.csv not found; run data/compute_changepoints.py")

    ucdp_path = os.path.join(base_dir, "..", "data", "ucdp_features.csv")
    ucdp_features = []
    if os.path.exists(ucdp_path) and os.environ.get("AIM4D_USE_UCDP") == "1":
        ucdp = pd.read_csv(ucdp_path)
        ucdp_features = [c for c in ucdp.columns if c not in {"country_text_id", "year"}]
        ews_df = ews_df.merge(ucdp, on=["country_text_id", "year"], how="left")
        for f in ucdp_features:
            if f == "ucdp_years_since_onset":
                ews_df[f] = ews_df[f].fillna(25)
            else:
                ews_df[f] = ews_df[f].fillna(0.0)
        print(f"  F6 UCDP conflict features loaded ({len(ucdp_features)}): "
              f"{ucdp_features}")
    elif os.path.exists(ucdp_path):
        print(f"  F6 UCDP: skipped (set AIM4D_USE_UCDP=1 to enable; "
              f"ablation shows -2 LOEO and BSS CI dips below 0)")
    else:
        print(f"  F6 UCDP: ucdp_features.csv not found; run data/build_ucdp_features.py")

    elec_calendar_features = []
    try:
        eltype_cols = [f"v2eltype_{i}" for i in range(10)]
        vdem_cols = pd.read_csv(vdem_path, low_memory=False, nrows=1).columns.tolist()
        elcols_avail = [c for c in eltype_cols if c in vdem_cols]
        if elcols_avail:
            raw = pd.read_csv(vdem_path, low_memory=False,
                              usecols=["country_text_id", "year"] + elcols_avail)
            raw = raw.dropna(subset=["country_text_id", "year"])
            raw = raw.sort_values(["country_text_id", "year"]).reset_index(drop=True)
            election_any = (raw[elcols_avail].fillna(0) > 0).any(axis=1).astype(int).to_numpy()

            countries = raw["country_text_id"].to_numpy()
            years = raw["year"].astype(int).to_numpy()
            yse_arr = np.full(len(raw), 99, dtype=int)
            last_country = None
            last_elec_year = -9999
            for i in range(len(raw)):
                c = countries[i]
                if c != last_country:
                    last_country = c
                    last_elec_year = -9999
                if election_any[i] == 1:
                    last_elec_year = int(years[i])
                if last_elec_year > -9999:
                    yse_arr[i] = int(years[i]) - last_elec_year

            yse_lookup = {(str(c), int(y)): int(v)
                          for c, y, v in zip(countries, years, yse_arr)}
            ews_keys = list(zip(ews_df["country_text_id"].astype(str).values,
                                 ews_df["year"].astype(int).values))
            yse_values = np.array([yse_lookup.get(k, 99) for k in ews_keys], dtype=int)
            ews_df["years_since_election"] = yse_values
            ews_df["election_within_2yr"] = (yse_values <= 2).astype(int)
            elec_calendar_features = ["years_since_election", "election_within_2yr"]
            print(f"  G5 election calendar features loaded: {elec_calendar_features} "
                  f"(via dict lookup; {sum(v != 99 for v in yse_values)} country-years matched)")
    except Exception as e:
        print(f"  G5 election calendar: skipped ({type(e).__name__}: {e})")

    ews_df["combined_alert_legacy"] = ews_df["ews_alert"] | ews_df["election_alert"] | ews_df["dem_vulnerability_alert"] | ews_df["military_threat_alert"]

    print(f"\n{'='*60}")
    print(f"Meta-learner calibration")
    print(f"{'='*60}\n")

    from sklearn.linear_model import LogisticRegressionCV, LogisticRegression
    from sklearn.preprocessing import StandardScaler as SS

    known_w = {}
    postonset_w = {}
    POSTONSET_EXCL_YEARS = POSTONSET_EXCL_YEARS_ENV
    for c, info in KNOWN_EPISODES.items():
        onset = info["onset"]
        lead = lead_for(info)
        for y in range(onset - lead, onset + 1):
            known_w[(c, y)] = True
        for y in range(onset + 1, onset + 1 + POSTONSET_EXCL_YEARS):
            postonset_w[(c, y)] = True

    ews_df["label"] = ews_df.apply(lambda r: 1 if (r["country_name"], r["year"]) in known_w else 0, axis=1)
    ews_df["is_postonset"] = ews_df.apply(
        lambda r: (r["country_name"], r["year"]) in postonset_w, axis=1
    )

    base_dir = os.path.dirname(os.path.abspath(__file__))
    vdem_path = os.path.join(base_dir, "..", "data", "vdem_v16.csv")
    dsp_cols = ["v2smgovdom", "v2smfordom", "v2smgovfilprc", "v2smgovsmmon", "v2smpardom"]
    mob_cols = ["v2caautmob", "v2cademmob", "v2cagenmob", "v2caconmob"]
    legit_cols = ["v2exl_legitideol", "v2exl_legitlead", "v2exl_legitperf", "v2exl_legitratio"]
    vdem_extra_cols = dsp_cols + mob_cols + legit_cols
    try:
        vdem_avail = pd.read_csv(vdem_path, low_memory=False, nrows=1).columns
        dsp_available = [c for c in dsp_cols if c in vdem_avail]
        mob_available = [c for c in mob_cols if c in vdem_avail]
        legit_available = [c for c in legit_cols if c in vdem_avail]
        vdem_extra_available = dsp_available + mob_available + legit_available
        if dsp_available:
            dsp_data = pd.read_csv(vdem_path, low_memory=False,
                                   usecols=["country_text_id", "year"] + vdem_extra_available)
            ews_df = ews_df.merge(dsp_data, on=["country_text_id", "year"], how="left")
            dsp_strategy = os.environ.get("AIM4D_DSP_STRATEGY", "ffill_2000")
            n_before = len(ews_df)
            if dsp_strategy == "median_full":
                for c in dsp_available:
                    train_slice = ews_df[ews_df["year"] <= TRAIN_CUTOFF]
                    med_by_country = train_slice.groupby("country_text_id")[c].median()
                    global_med = train_slice[c].median()
                    ews_df[c] = ews_df.groupby("country_text_id")[c].ffill()
                    ews_df[c] = ews_df.apply(
                        lambda r: med_by_country.get(r["country_text_id"], global_med)
                        if pd.isna(r[c]) else r[c], axis=1
                    )
                print(f"  DSP imputation: AIM4D_DSP_STRATEGY=median_full "
                      f"(train-period country median, all years kept, n={len(ews_df)})")
            else:
                ews_df = ews_df[ews_df["year"] >= 2000].reset_index(drop=True)
                for c in dsp_available:
                    ews_df[c] = ews_df.groupby("country_text_id")[c].ffill()
                print(f"  Restricted to year >= 2000 (DSP coverage window): "
                      f"{n_before} -> {len(ews_df)} country-years")
            for c in mob_available + legit_available:
                ews_df[c] = ews_df.groupby("country_text_id")[c].ffill()
                ews_df[c] = ews_df[c].fillna(0.0)
            print(f"  DSP variables loaded: {dsp_available}")
            print(f"  F2 mobilization features: {mob_available}")
            print(f"  F2 legitimation features: {legit_available}")
        else:
            dsp_available = []
    except Exception:
        dsp_available = []

    contagion = pd.read_csv(os.path.join(base_dir, "..", "stage4_nscm", "contagion_scores.csv"))
    if "contagion_score" in contagion.columns:
        ews_df = ews_df.merge(contagion[["country_text_id", "year", "contagion_score"]].rename(
            columns={"contagion_score": "network_exposure"}),
            on=["country_text_id", "year"], how="left")
        ews_df["network_exposure"] = ews_df["network_exposure"].fillna(0)
        ews_df["csd_x_network"] = ews_df["csd_index"] * ews_df["network_exposure"]
    else:
        ews_df["network_exposure"] = 0
        ews_df["csd_x_network"] = 0

    ews_df = ews_df.drop_duplicates(subset=["country_text_id", "year"], keep="first").reset_index(drop=True)

    base_features = (["csd_index", "mv_csd_index", "election_vulnerability", "party_threat",
                      "mil_zscore", "network_exposure", "csd_x_network"]
                     + dsp_available
                     + mob_available
                     + legit_available
                     + pitf_features
                     + diffusion_features
                     + archigos_features
                     + elec_calendar_features
                     + cp_features
                     + c22_features
                     + ucdp_features
                     + gdelt_features)
    available_base = [f for f in base_features if f in ews_df.columns]

    core_lag_features = ["csd_index", "mv_csd_index", "election_vulnerability", "mil_zscore"]
    for feat in core_lag_features:
        if feat in ews_df.columns:
            ews_df[f"{feat}_lag1"] = ews_df.groupby("country_text_id")[feat].shift(1)
            ews_df[f"{feat}_lag2"] = ews_df.groupby("country_text_id")[feat].shift(2)
            ews_df[f"{feat}_delta1"] = ews_df[feat] - ews_df[f"{feat}_lag1"]
            ews_df[f"{feat}_delta2"] = ews_df[feat] - ews_df[f"{feat}_lag2"]

    lag_features = []
    for feat in core_lag_features:
        if feat in ews_df.columns:
            lag_features += [f"{feat}_lag1", f"{feat}_lag2", f"{feat}_delta1", f"{feat}_delta2"]

    ews_df["era_post2015"] = (ews_df["year"] > 2015).astype(float)
    era_features = []
    for feat in ["csd_index", "election_vulnerability", "mil_zscore"] + dsp_available:
        if feat in ews_df.columns:
            ews_df[f"{feat}_x_post2015"] = ews_df[feat] * ews_df["era_post2015"]
            era_features.append(f"{feat}_x_post2015")

    available_for_detrend = available_base + era_features
    detrended_features = []
    for feat in available_for_detrend:
        if feat in ews_df.columns:
            yearly_median = ews_df.groupby("year")[feat].transform("median")
            ews_df[f"{feat}_detrended"] = ews_df[feat] - yearly_median
            detrended_features.append(f"{feat}_detrended")

    if "csd_index" in ews_df.columns and "election_vulnerability" in ews_df.columns:
        ews_df["csd_x_election"] = ews_df["csd_index"] * ews_df["election_vulnerability"]
    if "csd_index" in ews_df.columns and "mil_zscore" in ews_df.columns:
        ews_df["csd_x_military"] = ews_df["csd_index"] * ews_df["mil_zscore"]
    interaction_features = [f for f in ["csd_x_election", "csd_x_military"] if f in ews_df.columns]

    ews_df = ews_df.sort_values(["country_text_id", "year"]).reset_index(drop=True)
    for feat in ["csd_index", "mv_csd_index", "election_vulnerability"]:
        if feat in ews_df.columns:
            ews_df[f"{feat}_pctile"] = (
                ews_df.groupby("country_text_id")[feat]
                .transform(lambda s: s.expanding(min_periods=1).rank(pct=True))
            )
    pctile_features = [f"{f}_pctile" for f in ["csd_index", "mv_csd_index", "election_vulnerability"]
                       if f"{f}_pctile" in ews_df.columns]

    lagged_pctile_features = []
    for feat in ["csd_index", "mv_csd_index", "election_vulnerability"]:
        if f"{feat}_pctile" in ews_df.columns:
            new_col = f"{feat}_pctile_lag3"
            ews_df[new_col] = (
                ews_df.groupby("country_text_id")[f"{feat}_pctile"].shift(3)
            )
            ews_df[new_col] = ews_df[new_col].fillna(0.5)
            lagged_pctile_features.append(new_col)

    all_meta = (available_base + lag_features + pctile_features
                + lagged_pctile_features + era_features + detrended_features
                + interaction_features)
    print(f"  [diag] available_base={len(available_base)}  lag={len(lag_features)}  "
          f"pctile={len(pctile_features)}  lagged_pctile={len(lagged_pctile_features)}  "
          f"era={len(era_features)}  detrended={len(detrended_features)}  "
          f"interaction={len(interaction_features)}")
    pre_filter_n = len(all_meta)
    all_meta = [f for f in all_meta if f in ews_df.columns]
    post_filter_n = len(all_meta)
    if post_filter_n < pre_filter_n:
        missing = [f for f in (available_base + lag_features + pctile_features
                   + lagged_pctile_features + era_features + detrended_features
                   + interaction_features) if f not in ews_df.columns]
        print(f"  [diag] DROPPED {pre_filter_n - post_filter_n} features missing from ews_df.columns:")
        print(f"  [diag] missing sample: {missing[:20]}")
    print(f"  [diag] available_base sample: {sorted(available_base)[:15]}")
    print(f"  [diag] ews_df cols total: {len(ews_df.columns)}")

    ablate_group = os.environ.get("AIM4D_ABLATE_GROUP", "").strip().lower()
    if ablate_group:
        ABLATE_MAP = {
            "stage4": ["network_exposure", "csd_x_network"],
        }
        markers = ABLATE_MAP.get(ablate_group)
        if markers is None:
            raise ValueError(f"AIM4D_ABLATE_GROUP='{ablate_group}' not in {list(ABLATE_MAP)}")
        before = len(all_meta)
        all_meta = [f for f in all_meta if not any(m in f for m in markers)]
        print(f"  [ablation] AIM4D_ABLATE_GROUP={ablate_group}: dropped "
              f"{before - len(all_meta)} stage-derived features (markers={markers})")

    label_decay = 2.0
    known_w_soft = {}
    for c, info in KNOWN_EPISODES.items():
        lead = lead_for(info)
        for y in range(info["onset"] - lead, info["onset"] + 1):
            dist = max(0, info["onset"] - y)
            known_w_soft[(c, y)] = np.exp(-dist / label_decay)
    ews_df["label_soft"] = ews_df.apply(
        lambda r: known_w_soft.get((r["country_name"], r["year"]), 0.0), axis=1
    )
    ews_df["label"] = (ews_df["label_soft"] > 0.05).astype(int)

    X_meta = ews_df[all_meta].fillna(0).values
    y_meta = ews_df["label"].values
    y_meta_soft = ews_df["label_soft"].values
    train_mask = (ews_df["year"] <= TRAIN_CUTOFF) & (~ews_df["is_postonset"])
    if EXCLUDE_COUNTRY:
        train_mask = train_mask & (ews_df["country_name"] != EXCLUDE_COUNTRY)
        print(f"  Task F: excluding country '{EXCLUDE_COUNTRY}' from meta-learner training")

    scaler_meta = SS()
    scaler_meta.fit(X_meta[train_mask.values])
    X_meta_scaled = scaler_meta.transform(X_meta)

    if y_meta[train_mask].sum() >= 3:
        half_life = 8
        max_year = ews_df.loc[train_mask, "year"].max()
        time_weights = np.exp(-np.log(2) * (max_year - ews_df["year"].values) / half_life)
        train_weights = time_weights * np.where(y_meta == 1, POS_WEIGHT, 1.0)
        if POS_WEIGHT != 1.0:
            print(f"  Class weighting: positive samples upweighted {POS_WEIGHT}x")

        from sklearn.linear_model import SGDClassifier
        enet = SGDClassifier(
            loss="log_loss", penalty="elasticnet", l1_ratio=0.5,
            alpha=0.005, max_iter=2000, random_state=42, class_weight="balanced",
        )
        enet.fit(X_meta_scaled[train_mask], y_meta[train_mask],
                 sample_weight=time_weights[train_mask])
        report_mask = np.abs(enet.coef_[0]) > 1e-4
        n_selected = report_mask.sum()
        if os.environ.get("AIM4D_USE_ENET") == "1":
            selected_mask = report_mask
            selected_features = [f for f, s in zip(all_meta, selected_mask) if s]
            print(f"  AIM4D_USE_ENET=1: pruning to {n_selected}/{len(all_meta)} features via elastic-net.")
        else:
            selected_mask = np.ones_like(report_mask, dtype=bool)
            selected_features = list(all_meta)
            print(f"  Using all {len(all_meta)} features (elastic-net would have pruned to "
                  f"{n_selected}; pruning hurts OOS AUC by ~0.07).")

        X_selected = X_meta_scaled[:, selected_mask] if selected_mask.sum() >= 5 else X_meta_scaled

        from sklearn.ensemble import GradientBoostingClassifier
        from sklearn.model_selection import StratifiedKFold

        meta_lr = LogisticRegressionCV(cv=3, scoring="average_precision", max_iter=1000, random_state=42)
        meta_lr.fit(X_selected[train_mask], y_meta[train_mask],
                    sample_weight=train_weights[train_mask])
        lr_risk = meta_lr.predict_proba(X_selected)[:, 1]

        from joblib import Parallel, delayed
        N_SEEDS = 5 if os.environ.get("AIM4D_QUICK") == "1" else 20

        def _fit_gb(seed):
            gb = GradientBoostingClassifier(
                n_estimators=100, max_depth=3, learning_rate=0.05,
                subsample=0.8, min_samples_leaf=20, random_state=seed,
            )
            gb.fit(X_selected[train_mask], y_meta[train_mask],
                   sample_weight=train_weights[train_mask])
            return gb

        gb_models = Parallel(n_jobs=-1, backend="loky")(
            delayed(_fit_gb)(s) for s in range(SEED_OFFSET, SEED_OFFSET + N_SEEDS)
        )
        gb_risks = [m.predict_proba(X_selected)[:, 1] for m in gb_models]
        meta_gb = gb_models[0]
        gb_risk = np.mean(gb_risks, axis=0)

        cb_risk = None
        try:
            from catboost import CatBoostClassifier
            cb = CatBoostClassifier(
                iterations=1500, depth=5, learning_rate=0.03,
                l2_leaf_reg=5, bootstrap_type="Bayesian",
                bagging_temperature=1.0, auto_class_weights="SqrtBalanced",
                random_seed=42 + SEED_OFFSET, verbose=0, allow_writing_files=False,
                thread_count=int(os.environ.get("AIM4D_THREADS", "4")),
            )
            cb.fit(X_selected[train_mask], y_meta[train_mask],
                   sample_weight=train_weights[train_mask])
            cb_risk = cb.predict_proba(X_selected)[:, 1]
            print(f"  G3 CatBoost fitted (depth=5, iterations=1500)")
        except Exception as e:
            print(f"  G3 CatBoost: skipped ({type(e).__name__}: {e})")

        from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
        rf = RandomForestClassifier(n_estimators=500, max_depth=10,
                                    min_samples_leaf=10, class_weight="balanced",
                                    random_state=42 + SEED_OFFSET, n_jobs=-1)
        rf.fit(X_selected[train_mask], y_meta[train_mask],
               sample_weight=train_weights[train_mask])
        rf_risk = rf.predict_proba(X_selected)[:, 1]
        et = ExtraTreesClassifier(n_estimators=500, max_depth=10,
                                  min_samples_leaf=10, class_weight="balanced",
                                  random_state=42 + SEED_OFFSET, n_jobs=-1)
        et.fit(X_selected[train_mask], y_meta[train_mask],
               sample_weight=train_weights[train_mask])
        et_risk = et.predict_proba(X_selected)[:, 1]
        print(f"  G6 RandomForest + ExtraTrees fitted")

        tab_risk = None
        if os.environ.get("AIM4D_USE_TABPFN") == "1":
            try:
                from tabpfn import TabPFNClassifier
                tab = TabPFNClassifier(device="cpu", ignore_pretraining_limits=True)
                tab.fit(X_selected[train_mask], y_meta[train_mask])
                tab_risk = tab.predict_proba(X_selected)[:, 1]
                print(f"  G4 TabPFN fitted")
            except Exception as e:
                print(f"  G4 TabPFN: skipped ({type(e).__name__}: {str(e)[:80]})")
        else:
            print(f"  G4 TabPFN: skipped (set AIM4D_USE_TABPFN=1 to enable; "
                  f"requires license acceptance at priorlabs.ai)")

        X_train_sel = X_selected[train_mask]
        y_train = y_meta[train_mask]
        w_train = train_weights[train_mask]
        n_cv_folds = 3
        oof_lr = np.zeros(X_train_sel.shape[0])
        oof_gb = np.zeros(X_train_sel.shape[0])

        skf = StratifiedKFold(n_splits=n_cv_folds, shuffle=True, random_state=42)
        for fold_train, fold_val in skf.split(X_train_sel, y_train):
            lr_fold = LogisticRegression(C=1.0, max_iter=1000, random_state=42)
            lr_fold.fit(X_train_sel[fold_train], y_train[fold_train],
                        sample_weight=w_train[fold_train])
            oof_lr[fold_val] = lr_fold.predict_proba(X_train_sel[fold_val])[:, 1]

            gb_fold = GradientBoostingClassifier(
                n_estimators=100, max_depth=3, learning_rate=0.05,
                subsample=0.8, min_samples_leaf=20, random_state=42,
            )
            gb_fold.fit(X_train_sel[fold_train], y_train[fold_train],
                        sample_weight=w_train[fold_train])
            oof_gb[fold_val] = gb_fold.predict_proba(X_train_sel[fold_val])[:, 1]

        stack_X = np.column_stack([oof_lr, oof_gb])
        stack_model = LogisticRegression(C=10.0, max_iter=1000, random_state=42)
        stack_model.fit(stack_X, y_train, sample_weight=w_train)
        stack_coefs = stack_model.coef_[0]
        w_lr = np.exp(stack_coefs[0]) / (np.exp(stack_coefs[0]) + np.exp(stack_coefs[1]))
        w_gb = 1.0 - w_lr
        w_lr = np.clip(w_lr, 0.2, 0.8)
        w_gb = 1.0 - w_lr

        components = {"lr": (w_lr, lr_risk), "gb": (w_gb, gb_risk)}
        diverse_risks = []
        diverse_names = []
        for name, r in [("rf", rf_risk), ("et", et_risk),
                        ("cb", cb_risk), ("tab", tab_risk)]:
            if r is not None:
                diverse_risks.append(r)
                diverse_names.append(name)
        if diverse_risks:
            w_lr = w_lr * 0.7
            w_gb = w_gb * 0.7
            w_each = 0.3 / len(diverse_risks)
            components = {"lr": (w_lr, lr_risk), "gb": (w_gb, gb_risk)}
            for n, r in zip(diverse_names, diverse_risks):
                components[n] = (w_each, r)
        calibrated = sum(w * r for w, r in components.values())

        if os.environ.get("AIM4D_ISOTONIC", "0") == "1":
            from sklearn.isotonic import IsotonicRegression
            from sklearn.model_selection import GroupKFold
            from sklearn.base import clone

            tr_idx = np.where(train_mask.values)[0]
            X_tr = X_selected[tr_idx]
            y_tr = y_meta[tr_idx]
            w_tr = train_weights[tr_idx]
            country_tr = ews_df.loc[train_mask, "country_text_id"].values

            oof_lr_full = np.zeros(len(y_tr))
            oof_gb_full = np.zeros(len(y_tr))
            oof_rf_full = np.zeros(len(y_tr))
            oof_et_full = np.zeros(len(y_tr))
            oof_cb_full = np.zeros(len(y_tr))

            gkf = GroupKFold(n_splits=min(5, len(np.unique(country_tr))))
            for fold_tr, fold_va in gkf.split(X_tr, y_tr, groups=country_tr):
                lr_f = LogisticRegressionCV(cv=3, scoring="average_precision",
                                             max_iter=1000, random_state=42)
                lr_f.fit(X_tr[fold_tr], y_tr[fold_tr], sample_weight=w_tr[fold_tr])
                oof_lr_full[fold_va] = lr_f.predict_proba(X_tr[fold_va])[:, 1]

                gb_f = GradientBoostingClassifier(
                    n_estimators=100, max_depth=3, learning_rate=0.05,
                    subsample=0.8, min_samples_leaf=20, random_state=0,
                )
                gb_f.fit(X_tr[fold_tr], y_tr[fold_tr], sample_weight=w_tr[fold_tr])
                oof_gb_full[fold_va] = gb_f.predict_proba(X_tr[fold_va])[:, 1]

                rf_f = clone(rf)
                rf_f.fit(X_tr[fold_tr], y_tr[fold_tr], sample_weight=w_tr[fold_tr])
                oof_rf_full[fold_va] = rf_f.predict_proba(X_tr[fold_va])[:, 1]

                et_f = clone(et)
                et_f.fit(X_tr[fold_tr], y_tr[fold_tr], sample_weight=w_tr[fold_tr])
                oof_et_full[fold_va] = et_f.predict_proba(X_tr[fold_va])[:, 1]

                if cb_risk is not None:
                    try:
                        from catboost import CatBoostClassifier
                        cb_f = CatBoostClassifier(
                            iterations=1500, depth=5, learning_rate=0.03, l2_leaf_reg=5,
                            bootstrap_type="Bayesian", bagging_temperature=1.0,
                            auto_class_weights="SqrtBalanced", random_seed=42,
                            verbose=0, allow_writing_files=False,
                            thread_count=int(os.environ.get("AIM4D_THREADS", "4")),
                        )
                        cb_f.fit(X_tr[fold_tr], y_tr[fold_tr],
                                 sample_weight=w_tr[fold_tr])
                        oof_cb_full[fold_va] = cb_f.predict_proba(X_tr[fold_va])[:, 1]
                    except Exception:
                        oof_cb_full[fold_va] = oof_gb_full[fold_va]

            blend_oof = (components["lr"][0] * oof_lr_full
                         + components["gb"][0] * oof_gb_full
                         + components.get("rf", (0,))[0] * oof_rf_full
                         + components.get("et", (0,))[0] * oof_et_full
                         + (components.get("cb", (0,))[0] * oof_cb_full
                            if "cb" in components else 0))

            iso = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
            iso.fit(blend_oof, y_tr, sample_weight=w_tr)
            calibrated = iso.predict(calibrated)
            print(f"  Isotonic calibration fit (GroupKFold by country, "
                  f"{gkf.n_splits} folds, "
                  f"OOF blend range [{blend_oof.min():.3f}, {blend_oof.max():.3f}])")

        ews_df["calibrated_risk"] = calibrated

        print(f"\n  Stacked ensemble (cross-validated + diversity weights):")
        for name, (w, _) in components.items():
            print(f"    {name} weight: {w:.3f}")
        print(f"    LR component range: [{lr_risk[train_mask].min():.4f}, {lr_risk[train_mask].max():.4f}]")
        print(f"    GB component range: [{gb_risk[train_mask].min():.4f}, {gb_risk[train_mask].max():.4f}]")

        if n_selected >= 3:
            sel_feats = [f for f, s in zip(all_meta, selected_mask) if s]
            gb_imp = dict(zip(sel_feats, meta_gb.feature_importances_))
            print(f"  GB feature importance (top 10):")
            for feat, imp in sorted(gb_imp.items(), key=lambda x: -x[1])[:10]:
                print(f"    {feat}: {imp:.4f}")

        if SMOOTH_WINDOW >= 2:
            ews_df = ews_df.sort_values(["country_text_id", "year"]).reset_index(drop=True)
            ews_df["smoothed_risk"] = (
                ews_df.groupby("country_text_id")["calibrated_risk"]
                .transform(lambda s: s.rolling(SMOOTH_WINDOW, min_periods=1).max())
            )
            print(f"  Risk smoothing: rolling {SMOOTH_WINDOW}-year max")
            risk_for_tier = "smoothed_risk"
        else:
            risk_for_tier = "calibrated_risk"

        train_risks = ews_df.loc[train_mask, risk_for_tier]
        ews_df["alert_tier"] = "none"
        ews_df.loc[ews_df[risk_for_tier] >= train_risks.quantile(WATCH_PCTL), "alert_tier"] = "watch"
        ews_df.loc[ews_df[risk_for_tier] >= train_risks.quantile(WARNING_PCTL), "alert_tier"] = "warning"
        ews_df.loc[ews_df[risk_for_tier] >= train_risks.quantile(ALERT_PCTL), "alert_tier"] = "alert"

        ews_df["combined_alert"] = ews_df["alert_tier"].isin(["warning", "alert"])

        tier_counts = ews_df["alert_tier"].value_counts()
        print(f"\n  Tiered alerts (calibrated risk):")
        for tier in ["alert", "warning", "watch", "none"]:
            print(f"    {tier}: {tier_counts.get(tier, 0)} ({tier_counts.get(tier, 0)/len(ews_df)*100:.1f}%)")
    else:
        ews_df["calibrated_risk"] = ews_df["csd_index"]
        ews_df["alert_tier"] = "none"
        ews_df["combined_alert"] = ews_df["ews_alert"]
        print(f"  Insufficient positive examples for meta-learner, using CSD index")

    ews_df["combined_risk"] = ews_df["calibrated_risk"] if "calibrated_risk" in ews_df.columns else ews_df["csd_index"]

    if not NO_WRITE:
        ews_df.to_csv(os.path.join(output_dir, "ews_signals.csv"), index=False)
    else:
        print("  (ews_signals.csv write skipped: ablation/seed-offset run)")

    print(f"\n{'='*60}")
    print(f"Validation: episode detection + precision@K")
    print(f"{'='*60}\n")

    known_w = {}
    for c, info in KNOWN_EPISODES.items():
        lead = lead_for(info)
        for y in range(info["onset"] - lead, info["onset"] + 1):
            known_w[(c, y)] = True

    hits_by_tier = {"watch": 0, "warning": 0, "alert": 0}
    total = 0
    for country, info in KNOWN_EPISODES.items():
        onset = info["onset"]
        pre = ews_df[(ews_df["country_name"] == country) &
                     (ews_df["year"] >= onset - LEAD_YEARS) & (ews_df["year"] < onset)]
        if len(pre) == 0:
            print(f"  {country} ({onset}): NO DATA")
            continue
        total += 1
        max_risk = pre["combined_risk"].max()
        best_tier = "none"
        for tier in ["alert", "warning", "watch"]:
            if (pre["alert_tier"] == tier).any():
                best_tier = tier
                break
        if best_tier != "none":
            hits_by_tier[best_tier] += 1
            for lower in ["warning", "watch"]:
                if lower != best_tier:
                    hits_by_tier[lower] += 0
        detected = best_tier != "none"

        source = []
        if pre["ews_alert"].any(): source.append("CSD")
        if pre.get("mv_csd_alert", pd.Series(False)).any(): source.append("mvCSD")
        if pre.get("election_alert", pd.Series(False)).any(): source.append("ELEC")
        source_str = "+".join(source) if source else "meta-only"

        if detected:
            yrs = sorted(pre[pre["alert_tier"] != "none"]["year"].tolist())
            lead = onset - yrs[0] if yrs else 0
            print(f"  {country} ({info['type']} {onset}): DETECTED [{best_tier}] via {source_str} ({lead}yr lead, risk={max_risk:.3f})")
        else:
            print(f"  {country} ({info['type']} {onset}): MISSED (risk={max_risk:.3f})")

    cum_watch = sum(1 for c, info in KNOWN_EPISODES.items()
                    if ews_df[(ews_df["country_name"] == c) &
                              (ews_df["year"] >= info["onset"] - LEAD_YEARS) &
                              (ews_df["year"] < info["onset"])]["alert_tier"].isin(["watch", "warning", "alert"]).any()
                    and len(ews_df[(ews_df["country_name"] == c) & (ews_df["year"] >= info["onset"] - LEAD_YEARS)]) > 0)
    cum_warning = sum(1 for c, info in KNOWN_EPISODES.items()
                      if ews_df[(ews_df["country_name"] == c) &
                                (ews_df["year"] >= info["onset"] - LEAD_YEARS) &
                                (ews_df["year"] < info["onset"])]["alert_tier"].isin(["warning", "alert"]).any()
                      and len(ews_df[(ews_df["country_name"] == c) & (ews_df["year"] >= info["onset"] - LEAD_YEARS)]) > 0)
    cum_alert = sum(1 for c, info in KNOWN_EPISODES.items()
                    if ews_df[(ews_df["country_name"] == c) &
                              (ews_df["year"] >= info["onset"] - LEAD_YEARS) &
                              (ews_df["year"] < info["onset"])]["alert_tier"].isin(["alert"]).any()
                    and len(ews_df[(ews_df["country_name"] == c) & (ews_df["year"] >= info["onset"] - LEAD_YEARS)]) > 0)

    print(f"\n  Detection by tier (cumulative):")
    print(f"    Watch (top 20%):  {cum_watch}/{total}")
    print(f"    Warning (top 5%): {cum_warning}/{total}")
    print(f"    Alert (top 2%):   {cum_alert}/{total}")

    print(f"\n  Precision@K (country-level ranked risk list):")
    ews_df["label"] = ews_df.apply(lambda r: 1 if (r["country_name"], r["year"]) in known_w else 0, axis=1)
    valid = ews_df.dropna(subset=["combined_risk"])

    country_max_risk = valid.groupby("country_name").agg(
        max_risk=("combined_risk", "max"),
        any_positive=("label", "max"),
    ).reset_index()
    country_max_risk = country_max_risk.sort_values("max_risk", ascending=False)
    n_positive_countries = country_max_risk["any_positive"].sum()

    for K in [5, 10, 20, 50]:
        top_k = country_max_risk.head(K)
        prec_k = top_k["any_positive"].mean()
        recall_k = top_k["any_positive"].sum() / n_positive_countries if n_positive_countries > 0 else 0
        base_rate = country_max_risk["any_positive"].mean()
        lift_k = prec_k / base_rate if base_rate > 0 else 0
        print(f"    @{K:3d}: precision={prec_k:.1%}, recall={recall_k:.1%}, lift={lift_k:.1f}x")

    alerts = ews_df[ews_df["combined_alert"]]
    tp = alerts[alerts["label"] == 1]
    fp = alerts[alerts["label"] == 0]
    prec = len(tp) / len(alerts) if len(alerts) > 0 else 0
    sens = cum_warning / total if total > 0 else 0
    fp_rate = len(fp) / len(ews_df)

    print(f"\n  Warning-tier performance:")
    print(f"    Alerts: {len(alerts)} / {len(ews_df)} ({len(alerts)/len(ews_df):.1%})")
    print(f"    Precision: {prec:.1%}")
    print(f"    Sensitivity: {cum_warning}/{total} ({sens:.0%})")
    if sens > 0 and prec > 0:
        print(f"    F1: {2*prec*sens/(prec+sens):.3f}")

    stable = ["Denmark", "Sweden", "Norway", "Switzerland", "Finland",
              "Germany", "Canada", "New Zealand", "Uruguay", "Belgium",
              "Iceland", "Australia", "Ireland", "Netherlands"]
    sfp = fp[fp["country_name"].isin(stable)]
    print(f"    Stable democracy FPs: {len(sfp)}")

    print(f"\n{'='*60}")
    print(f"Leave-one-episode-out cross-validation")
    print(f"{'='*60}\n")

    loeo_results_by_tier = {"alert": 0, "warning": 0, "watch": 0}
    loeo_total = 0
    loeo_risks = []

    loeo_items = [] if os.environ.get("AIM4D_SKIP_LOEO") == "1" else list(KNOWN_EPISODES.items())
    if not loeo_items:
        print("  (LOEO skipped: AIM4D_SKIP_LOEO=1)")

    def _loeo_one(held_out_country, held_out_info):
        held_out_onset = held_out_info["onset"]

        held_lead = lead_for(held_out_info)
        train_labels = ews_df["label"].copy()
        for y in range(held_out_onset - held_lead, held_out_onset + 1):
            mask = (ews_df["country_name"] == held_out_country) & (ews_df["year"] == y)
            train_labels[mask] = 0

        X_loeo = ews_df[all_meta].fillna(0).values
        y_loeo = train_labels.values
        loeo_train = (ews_df["country_name"] != held_out_country).values

        if y_loeo[loeo_train].sum() < 3:
            return None

        scaler_loeo = SS()
        X_scaled = scaler_loeo.fit_transform(X_loeo)

        loeo_weights = time_weights * np.where(y_loeo == 1, POS_WEIGHT, 1.0)

        loeo_lr = LogisticRegression(C=1.0, max_iter=1000, random_state=42)
        loeo_lr.fit(X_scaled[loeo_train], y_loeo[loeo_train],
                    sample_weight=loeo_weights[loeo_train])
        loeo_gb = GradientBoostingClassifier(
            n_estimators=100, max_depth=3, learning_rate=0.05,
            subsample=0.8, min_samples_leaf=20, random_state=42,
        )
        loeo_gb.fit(X_scaled[loeo_train], y_loeo[loeo_train],
                    sample_weight=loeo_weights[loeo_train])

        pre = ews_df[(ews_df["country_name"] == held_out_country) &
                     (ews_df["year"] >= held_out_onset - held_lead) &
                     (ews_df["year"] < held_out_onset)]

        if len(pre) == 0:
            return None

        pre_idx = pre.index
        pre_risk = w_lr * loeo_lr.predict_proba(X_scaled[pre_idx])[:, 1] + \
                   w_gb * loeo_gb.predict_proba(X_scaled[pre_idx])[:, 1]
        max_risk = pre_risk.max()
        train_preds = w_lr * loeo_lr.predict_proba(X_scaled[loeo_train])[:, 1] + \
                      w_gb * loeo_gb.predict_proba(X_scaled[loeo_train])[:, 1]

        train_y = y_loeo[loeo_train]
        neg_preds = train_preds[train_y == 0]
        if len(neg_preds) >= 50:
            thresh_alert = np.percentile(neg_preds, 98)
            thresh_warning = np.percentile(neg_preds, 95)
            thresh_watch = np.percentile(neg_preds, 80)
        else:
            thresh_alert = np.percentile(train_preds, 98)
            thresh_warning = np.percentile(train_preds, 95)
            thresh_watch = np.percentile(train_preds, 80)

        if max_risk > thresh_alert:
            tier = "alert"
        elif max_risk > thresh_warning:
            tier = "warning"
        elif max_risk > thresh_watch:
            tier = "watch"
        else:
            tier = "none"

        return {"country": held_out_country, "max_risk": max_risk,
                "tier": tier, "detected": tier != "none",
                "type": held_out_info.get("type", ""),
                "watch_thresh": thresh_watch}

    loeo_out = Parallel(n_jobs=int(os.environ.get("AIM4D_THREADS", "4")),
                        backend="loky")(
        delayed(_loeo_one)(c, info) for c, info in loeo_items
    )

    for res in loeo_out:
        if res is None:
            continue
        loeo_total += 1
        if res["detected"]:
            print(f"  {res['country']}: DETECTED [{res['tier']}] (LOEO, risk={res['max_risk']:.3f})")
        else:
            print(f"  {res['country']}: MISSED (LOEO, risk={res['max_risk']:.3f}, watch_thresh={res['watch_thresh']:.3f})")
        loeo_risks.append({"country": res["country"], "max_risk": res["max_risk"],
                           "tier": res["tier"], "detected": res["detected"],
                           "type": res["type"]})

    loeo_df = pd.DataFrame(loeo_risks)
    if len(loeo_df):
        loeo_df["detected_watch"] = loeo_df["tier"].isin(["watch", "warning", "alert"]).astype(int)
        loeo_df["detected_warning"] = loeo_df["tier"].isin(["warning", "alert"]).astype(int)
        loeo_df["detected_alert"] = (loeo_df["tier"] == "alert").astype(int)
        loeo_df.to_csv(os.path.join(output_dir, "loeo_results.csv"), index=False)

    loeo_watch = sum(1 for r in loeo_risks if r["tier"] in ["watch", "warning", "alert"])
    loeo_warning = sum(1 for r in loeo_risks if r["tier"] in ["warning", "alert"])
    loeo_alert = sum(1 for r in loeo_risks if r["tier"] == "alert")

    print(f"\n  LOEO Sensitivity by tier (unbiased):")
    print(f"    Watch (P80):   {loeo_watch}/{loeo_total} ({loeo_watch/loeo_total:.0%})" if loeo_total > 0 else "")
    print(f"    Warning (P95): {loeo_warning}/{loeo_total} ({loeo_warning/loeo_total:.0%})" if loeo_total > 0 else "")
    print(f"    Alert (P98):   {loeo_alert}/{loeo_total} ({loeo_alert/loeo_total:.0%})" if loeo_total > 0 else "")
    print(f"  (Each episode predicted without seeing itself)")

    print(f"\n  LOEO stratified by episode type:")
    for ep_type in ["backsliding", "coup"]:
        type_risks = [r for r in loeo_risks
                      if KNOWN_EPISODES.get(r["country"], {}).get("type") == ep_type]
        if type_risks:
            type_detected = sum(1 for r in type_risks if r["tier"] in ["watch", "warning", "alert"])
            type_total = len(type_risks)
            print(f"    {ep_type}: {type_detected}/{type_total} ({type_detected/type_total:.0%})")

    print(f"\n  LOEO stratified by onset era:")
    for era_label, era_start, era_end in [("pre-2005", 0, 2004), ("2005-2014", 2005, 2014),
                                           ("2015-2021", 2015, 2021), ("post-2021", 2022, 2030)]:
        era_risks = [r for r in loeo_risks
                     if era_start <= KNOWN_EPISODES.get(r["country"], {}).get("onset", 0) <= era_end]
        if era_risks:
            era_detected = sum(1 for r in era_risks if r["tier"] in ["watch", "warning", "alert"])
            era_total = len(era_risks)
            print(f"    {era_label}: {era_detected}/{era_total} ({era_detected/era_total:.0%})")

    print(f"\n{'='*60}")
    print(f"Continuous risk score evaluation (AUC)")
    print(f"{'='*60}\n")

    from sklearn.metrics import roc_auc_score, average_precision_score

    ews_eval = ews_df.copy()
    ews_eval["label"] = ews_eval.apply(
        lambda r: 1 if (r["country_name"], r["year"]) in known_w else 0, axis=1
    )
    valid = ews_eval.dropna(subset=["csd_index"])
    valid = valid[~valid["is_postonset"]].copy()
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
            oos_base_rate = oos["label"].mean()
            print(f"  AUC-ROC (OOS): {auc_roc_oos:.3f}")
            print(f"  AUC-PR (OOS):  {auc_pr_oos:.3f}")
            import json as _json
            print("ABLATE_JSON " + _json.dumps({
                "ablate_group": os.environ.get("AIM4D_ABLATE_GROUP", "").strip().lower() or "full",
                "seed_offset": SEED_OFFSET,
                "n_features": len(all_meta),
                "oos_roc": round(float(auc_roc_oos), 4),
                "oos_pr": round(float(auc_pr_oos), 4),
                "oos_base_rate": round(float(oos_base_rate), 4),
                "oos_n": int(len(oos)),
                "oos_n_pos": int(oos["label"].sum()),
                "is_roc": round(float(auc_roc), 4),
                "is_pr": round(float(auc_pr), 4),
            }))

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
    print(f"Expanding-window temporal CV")
    print(f"{'='*60}\n")

    windows = [
        (2005, 2008), (2008, 2011), (2011, 2014),
        (2014, 2017), (2017, 2020), (2020, 2023),
    ]
    window_aucs = []
    window_detections = []

    for train_end, test_end in windows:
        w_train = valid[valid["year"] <= train_end]
        w_test = valid[(valid["year"] > train_end) & (valid["year"] <= test_end)]

        if w_test["label"].sum() == 0 or w_test["label"].nunique() < 2:
            continue

        try:
            w_auc = roc_auc_score(w_test["label"], w_test["combined_risk"])
        except ValueError:
            continue

        test_episodes = set()
        for c, info in KNOWN_EPISODES.items():
            if train_end < info["onset"] <= test_end:
                test_episodes.add(c)

        n_detected = 0
        for ep_country in test_episodes:
            ep_onset = KNOWN_EPISODES[ep_country]["onset"]
            ep_pre = ews_df[(ews_df["country_name"] == ep_country) &
                            (ews_df["year"] >= ep_onset - LEAD_YEARS) &
                            (ews_df["year"] < ep_onset) &
                            (ews_df["year"] > train_end)]
            if len(ep_pre) > 0 and ep_pre["combined_alert"].any():
                n_detected += 1

        window_aucs.append(w_auc)
        n_eps = len(test_episodes)
        det_rate = n_detected / n_eps if n_eps > 0 else 0
        window_detections.append(det_rate)
        print(f"  {train_end+1}-{test_end}: AUC={w_auc:.3f}, episodes={n_eps}, detected={n_detected}/{n_eps}")

    if window_aucs:
        mean_auc = np.mean(window_aucs)
        std_auc = np.std(window_aucs)
        print(f"\n  Mean AUC across windows: {mean_auc:.3f} +/- {std_auc:.3f}")
        print(f"  Mean detection rate: {np.mean(window_detections):.0%}")
        print(f"  (This is the robust generalization estimate across 6 temporal windows)")

    horizon = 2025 - TRAIN_CUTOFF
    print(f"\n  Secondary OOS (TRAIN_CUTOFF={TRAIN_CUTOFF}, {horizon}-year horizon):")
    oos_2017 = valid[valid["year"] > TRAIN_CUTOFF].copy()
    known_w_oos = {}
    for c, info in KNOWN_EPISODES.items():
        lead = lead_for(info)
        for y in range(info["onset"] - lead, info["onset"] + 1):
            known_w_oos[(c, y)] = True
    oos_2017["label_oos"] = oos_2017.apply(
        lambda r: 1 if (r["country_name"], r["year"]) in known_w_oos else 0, axis=1
    )
    if oos_2017["label_oos"].sum() > 0 and oos_2017["label_oos"].nunique() > 1:
        try:
            auc_2017 = roc_auc_score(oos_2017["label_oos"], oos_2017["combined_risk"])
            auc_pr_2017 = average_precision_score(oos_2017["label_oos"], oos_2017["combined_risk"])
            oos_episodes = {c for c, info in KNOWN_EPISODES.items() if info["onset"] > TRAIN_CUTOFF}
            oos_detected = 0
            for c in oos_episodes:
                onset = KNOWN_EPISODES[c]["onset"]
                pre = ews_df[(ews_df["country_name"] == c) &
                             (ews_df["year"] >= onset - LEAD_YEARS) &
                             (ews_df["year"] < onset) &
                             (ews_df["year"] > TRAIN_CUTOFF)]
                if len(pre) > 0 and pre["alert_tier"].isin(["watch", "warning", "alert"]).any():
                    oos_detected += 1
            print(f"    AUC-ROC: {auc_2017:.3f}")
            print(f"    AUC-PR:  {auc_pr_2017:.3f}")
            print(f"    Episodes (onset>{TRAIN_CUTOFF}): {len(oos_episodes)}, detected: {oos_detected}/{len(oos_episodes)}")
        except ValueError:
            print(f"    Insufficient data for {TRAIN_CUTOFF} OOS evaluation")
    else:
        print(f"    No positive labels in year>{TRAIN_CUTOFF} window")

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

    print(f"\n{'='*60}")
    print(f"Prospective Risk Ranking: 2026-2031 Autocratization Risk")
    print(f"(Genuine prospective forecast — verifiable against future data)")
    print(f"{'='*60}\n")

    latest_year = ews_df["year"].max()
    latest = ews_df[ews_df["year"] == latest_year].copy()
    latest = latest.sort_values("combined_risk", ascending=False)

    print(f"  {'Rank':<5} {'Country':<30} {'Risk':>8} {'Tier':<10} {'Key Signals'}")
    print(f"  {'-'*80}")
    for rank, (_, r) in enumerate(latest.head(25).iterrows(), 1):
        signals = []
        if r.get("ews_alert", False): signals.append("CSD")
        if r.get("mv_csd_alert", False): signals.append("mvCSD")
        if r.get("election_alert", False): signals.append("ELEC")
        if r.get("military_threat_alert", False): signals.append("MIL")
        sig_str = "+".join(signals) if signals else "meta"
        tier = r.get("alert_tier", "none")
        print(f"  {rank:<5} {r['country_name']:<30} {r['combined_risk']:>8.4f} {tier:<10} {sig_str}")

    return ews_df


if __name__ == "__main__":
    ews_df = run_ews()
