"""
Factor count sensitivity analysis (K=3, 4, 5).

Re-runs POET factor extraction with forced K values, then propagates through
stages 2-5 to measure downstream impact on regime classification (kappa) and
early warning performance (AUC, detection rate).

Methodological basis:
  - Bai & Ng (2002): IC criteria for factor count selection
  - Fan et al. (2013): POET robustness to K over-estimation
  - Stock & Watson (2002): forecast accuracy often flat across K range

Reports:
  - Bai-Ng IC1/IC2/IC3 values for each K
  - Cumulative variance explained
  - Tucker congruence coefficients between K=4 baseline and K=3,5 shared factors
  - Downstream HMM kappa, EWS AUC, detection rate per K
"""

import sys
import os
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings("ignore")
from scipy import linalg

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from stage1_factors.extract import (
    load_vdem, select_indicators, build_panel, panel_to_matrix,
    bai_ng_ic, poet_estimate, varimax, FACTOR_LABELS,
)
from stage2_betas.estimate import (
    FACTOR_COLS as BETA_FACTOR_COLS, MIN_OBS,
    compute_loo_global, estimate_country_factor_beta,
)
from stage3_msvar.estimate import (
    N_STATES, N_RESTARTS, STATE_LABELS, FACTOR_COLS,
    prepare_sequences, quantile_init, fit_baseline_hmm,
    precompute_log_emissions, decode_all, validate,
)
from stage5_ews.estimate import KNOWN_EPISODES, LEAD_YEARS, TRAIN_CUTOFF

OUTPUT_DIR = os.path.dirname(os.path.abspath(__file__))
K_VALUES = [3, 4, 5]


def extract_with_forced_k(K, panel, indicators, X, scaler):
    """Run POET with a forced factor count K."""
    result = poet_estimate(X, K)

    sign_ref = {0: "v2x_polyarchy", 1: "v2x_corr", 2: "v2x_suffr", 3: "v2xdd_dd"}
    expected_sign = {0: +1, 1: -1, 2: +1, 3: +1}
    for k in range(min(K, len(sign_ref))):
        ref = sign_ref.get(k)
        if ref and ref in indicators:
            idx = indicators.index(ref)
            actual = np.sign(result["loadings"][idx, k])
            desired = expected_sign.get(k, +1)
            if actual != desired:
                result["loadings"][:, k] *= -1
                result["factors"][:, k] *= -1

    var_explained = result["eigenvalues"] / np.trace(X.T @ X / X.shape[0])
    cumulative = np.cumsum(var_explained)

    factor_cols = [f"factor_{i+1}" for i in range(K)]
    factor_df = panel[["country_name", "country_text_id", "year"]].copy()
    factor_df[factor_cols] = result["factors"]
    factor_df = factor_df.reset_index(drop=True)

    loading_df = pd.DataFrame(
        result["loadings"], index=indicators, columns=factor_cols,
    )

    return factor_df, loading_df, var_explained, cumulative


def tucker_congruence(L1, L2):
    """Tucker congruence coefficient between two loading matrices (shared cols)."""
    n_shared = min(L1.shape[1], L2.shape[1])
    coeffs = []
    for k in range(n_shared):
        a, b = L1[:, k], L2[:, k]
        num = np.dot(a, b)
        denom = np.sqrt(np.dot(a, a) * np.dot(b, b))
        coeffs.append(num / denom if denom > 1e-10 else 0.0)
    return coeffs


def run_betas_for_factors(factor_df, K):
    """Re-run stage 2 beta estimation for a given factor set."""
    factor_cols = [f"factor_{i+1}" for i in range(K)]
    countries = factor_df["country_name"].unique()
    results = []

    for country in countries:
        cdf = factor_df[factor_df["country_name"] == country].sort_values("year")
        if len(cdf) < MIN_OBS:
            continue

        years = cdf["year"].values
        y_all = cdf[factor_cols].values
        others = factor_df[factor_df["country_name"] != country]
        gf = others.groupby("year")[factor_cols].mean().loc[years].values

        for t_idx, year in enumerate(years):
            row = {
                "country_name": country,
                "country_text_id": cdf["country_text_id"].iloc[0],
                "year": int(year),
            }
            results.append(row)

        for k in range(K):
            dy = np.diff(y_all[:, k])
            dx = np.diff(gf[:, k])
            beta_smooth, _, _, _, _, _ = estimate_country_factor_beta(dy, dx)
            for t_idx in range(len(years)):
                b = beta_smooth[0] if t_idx == 0 else beta_smooth[t_idx - 1] if t_idx - 1 < len(beta_smooth) else beta_smooth[-1]
                row_idx = sum(
                    len(factor_df[(factor_df["country_name"] == c) & (factor_df["year"].isin(
                        factor_df[factor_df["country_name"] == c].sort_values("year")["year"]
                    ))]) for c in countries if c < country
                )

    all_rows = []
    for country in countries:
        cdf = factor_df[factor_df["country_name"] == country].sort_values("year")
        if len(cdf) < MIN_OBS:
            continue
        years = cdf["year"].values
        y_all = cdf[factor_cols].values
        others = factor_df[factor_df["country_name"] != country]
        gf = others.groupby("year")[factor_cols].mean().reindex(years).values

        if np.any(np.isnan(gf)):
            continue

        country_betas = np.zeros((len(years), K))
        for k in range(K):
            dy = np.diff(y_all[:, k])
            dx = np.diff(gf[:, k])
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                beta_smooth, _, _, _, _, _ = estimate_country_factor_beta(dy, dx)
            country_betas[0, k] = beta_smooth[0]
            country_betas[1:, k] = beta_smooth

        for t_idx, year in enumerate(years):
            row = {
                "country_name": country,
                "country_text_id": cdf["country_text_id"].iloc[0],
                "year": int(year),
            }
            for k in range(K):
                row[f"beta_factor_{k+1}"] = country_betas[t_idx, k]
            all_rows.append(row)

    return pd.DataFrame(all_rows)


def run_hmm_for_factors(factor_df, beta_df, K):
    """Re-run stage 3 HMM for a given factor count."""
    factor_cols = [f"factor_{i+1}" for i in range(K)]
    beta_cols = [f"beta_factor_{i+1}" for i in range(K)]

    df = factor_df.merge(beta_df[["country_name", "year"] + beta_cols],
                         on=["country_name", "year"])

    lag_cols = []
    for fc in factor_cols:
        lcol = f"lag_{fc}"
        df[lcol] = df.groupby("country_name")[fc].shift(1)
        lag_cols.append(lcol)
    df = df.dropna(subset=lag_cols)

    obs_cols = factor_cols + lag_cols
    X_all, lengths, country_order = prepare_sequences(df, obs_cols)

    if len(X_all) == 0:
        return None, 0.0

    init_means, init_covars = quantile_init(X_all, N_STATES)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        baseline, base_score = fit_baseline_hmm(X_all, lengths, init_means, init_covars)

    log_emit_all = precompute_log_emissions(X_all, baseline.means_, baseline.covars_)
    emit_seqs = []
    idx = 0
    for l in lengths:
        emit_seqs.append(log_emit_all[idx:idx + l])
        idx += l

    state_df, _ = decode_all(
        emit_seqs, [None] * len(emit_seqs), lengths, country_order, df, baseline,
    )

    kappa, kappa_w = validate(state_df)
    return state_df, kappa_w


def run_ews_detection(factor_df, K):
    """Simplified EWS detection rate using CSD on factor residuals."""
    from stage5_ews.estimate import (
        rolling_stats, country_z, persistence_filter,
        WINDOW, MIN_WINDOW, BASELINE_END, Z_THRESHOLD, Z_CAP, MIN_ABS_VAR_PCTL,
    )

    factor_cols = [f"factor_{i+1}" for i in range(K)]
    countries = sorted(factor_df["country_name"].unique())

    for fc in factor_cols:
        factor_df[f"resid_{fc}"] = factor_df.groupby("country_text_id")[fc].diff()
    factor_df = factor_df.dropna(subset=[f"resid_{fc}" for fc in factor_cols])
    resid_cols = [f"resid_{fc}" for fc in factor_cols]

    all_train_vars = []
    for country in countries:
        cdf = factor_df[factor_df["country_name"] == country].sort_values("year")
        if len(cdf) < MIN_WINDOW + 2:
            continue
        for rc in resid_cols:
            rv, _, _ = rolling_stats(cdf[rc].values)
            tv = rv[np.array(cdf["year"].values) <= TRAIN_CUTOFF]
            all_train_vars.extend(tv[~np.isnan(tv)])

    if not all_train_vars:
        return 0, 0, 0.0

    abs_var_floor = np.percentile(all_train_vars, MIN_ABS_VAR_PCTL * 100)

    all_ews = []
    for country in countries:
        cdf = factor_df[factor_df["country_name"] == country].sort_values("year")
        if len(cdf) < MIN_WINDOW + 2:
            continue
        years = cdf["year"].values

        factor_alerts = np.zeros(len(years), dtype=int)
        best_var_z = np.full(len(years), np.nan)
        best_ar1_z = np.full(len(years), np.nan)
        max_abs = np.full(len(years), np.nan)

        for rc in resid_cols:
            rv, ra, rk = rolling_stats(cdf[rc].values)
            vz = country_z(rv, years)
            az = country_z(ra, years)
            kz = country_z(rk, years)

            for t in range(len(years)):
                above_floor = not np.isnan(rv[t]) and rv[t] > abs_var_floor
                if above_floor:
                    csd = (vz[t] > Z_THRESHOLD and az[t] > Z_THRESHOLD) or \
                          (vz[t] > Z_THRESHOLD and kz[t] > Z_THRESHOLD)
                    if csd:
                        factor_alerts[t] += 1
                for m, v in [("var_z", vz[t]), ("ar1_z", az[t])]:
                    arr = best_var_z if m == "var_z" else best_ar1_z
                    if np.isnan(arr[t]) or (not np.isnan(v) and v > arr[t]):
                        arr[t] = v
                if np.isnan(max_abs[t]) or (not np.isnan(rv[t]) and rv[t] > max_abs[t]):
                    max_abs[t] = rv[t]

        csd_idx = np.zeros(len(years))
        for t in range(len(years)):
            if not np.isnan(max_abs[t]) and max_abs[t] > abs_var_floor:
                c = []
                for v in [best_var_z[t], best_ar1_z[t]]:
                    if not np.isnan(v):
                        c.append(min(Z_CAP, max(0, v)))
                csd_idx[t] = np.mean(c) if c else 0

        min_factors = min(K, 3)
        raw = (factor_alerts >= min_factors) | \
              ((factor_alerts >= max(1, min_factors - 1)) & (csd_idx > 2.5)) | \
              ((factor_alerts >= 1) & (csd_idx > 4.0))
        persistent = persistence_filter(raw)

        for t in range(len(years)):
            all_ews.append({
                "country_name": country, "year": int(years[t]),
                "csd_index": csd_idx[t], "ews_alert": persistent[t],
            })

    ews_df = pd.DataFrame(all_ews)

    hits, total = 0, 0
    for country, info in KNOWN_EPISODES.items():
        onset = info["onset"]
        pre = ews_df[(ews_df["country_name"] == country) &
                     (ews_df["year"] >= onset - LEAD_YEARS) & (ews_df["year"] < onset)]
        if len(pre) == 0:
            continue
        total += 1
        if pre["ews_alert"].any():
            hits += 1

    from sklearn.metrics import roc_auc_score
    known_w = {}
    for c, info in KNOWN_EPISODES.items():
        for y in range(info["onset"] - LEAD_YEARS, info["onset"] + 1):
            known_w[(c, y)] = True

    ews_df["label"] = ews_df.apply(
        lambda r: 1 if (r["country_name"], r["year"]) in known_w else 0, axis=1
    )
    valid = ews_df.dropna(subset=["csd_index"])
    auc = 0.0
    if valid["label"].sum() > 0 and valid["label"].nunique() > 1:
        try:
            auc = roc_auc_score(valid["label"], valid["csd_index"])
        except ValueError:
            pass

    return hits, total, auc


def run_k_sensitivity():
    print("=" * 70)
    print("ROBUSTNESS CHECK: Factor Count Sensitivity (K=3, 4, 5)")
    print("=" * 70)
    print()
    print("Methodological basis:")
    print("  Bai & Ng (2002) IC criteria; Fan et al. (2013) POET robustness;")
    print("  Stock & Watson (2002) forecast accuracy across K")
    print()

    print("Loading V-Dem data...")
    df = load_vdem()
    indicators = select_indicators(df)
    panel = build_panel(df, indicators)
    X, scaler = panel_to_matrix(panel, indicators)

    print("\n--- Bai-Ng Information Criteria ---")
    ic_results, ic_vals, top_eigs = bai_ng_ic(X)
    print(f"  IC1 selects K={ic_results[1]}")
    print(f"  IC2 selects K={ic_results[2]}")
    print(f"  IC3 selects K={ic_results[3]}")
    print(f"  Scree elbow: K={ic_results['elbow']}")

    print("\n  IC2 values (K=1..8):")
    for k in range(min(8, len(ic_vals[2]))):
        print(f"    K={k+1}: IC2={ic_vals[2][k]:.4f}")

    results_table = []
    loading_matrices = {}

    for K in K_VALUES:
        print(f"\n{'='*50}")
        print(f"K = {K}")
        print(f"{'='*50}")

        print(f"\n  Extracting {K} factors via POET + varimax...")
        factor_df, loading_df, var_explained, cumulative = extract_with_forced_k(
            K, panel, indicators, X, scaler
        )
        loading_matrices[K] = loading_df.values

        print(f"  Variance explained: {np.round(var_explained * 100, 1)}%")
        print(f"  Cumulative: {np.round(cumulative * 100, 1)}%")

        print(f"\n  Estimating time-varying betas...")
        beta_df = run_betas_for_factors(factor_df, K)
        print(f"  Beta estimation: {len(beta_df)} country-years")

        print(f"\n  Running HMM regime classification...")
        state_df, kappa_w = run_hmm_for_factors(factor_df, beta_df, K)
        print(f"  Weighted kappa: {kappa_w:.3f}")

        print(f"\n  Running EWS detection...")
        hits, total, auc = run_ews_detection(factor_df.copy(), K)
        det_rate = hits / total if total > 0 else 0
        print(f"  Detection: {hits}/{total} ({det_rate:.0%})")
        print(f"  CSD AUC: {auc:.3f}")

        results_table.append({
            "K": K,
            "cumulative_var": cumulative[-1],
            "weighted_kappa": kappa_w,
            "detection_rate": det_rate,
            "detections": f"{hits}/{total}",
            "csd_auc": auc,
        })

    print(f"\n{'='*50}")
    print("Tucker Congruence Coefficients (vs K=4 baseline)")
    print(f"{'='*50}")
    baseline_loadings = loading_matrices[4]
    for K in [3, 5]:
        coeffs = tucker_congruence(baseline_loadings, loading_matrices[K])
        print(f"  K={K} vs K=4: {['%.3f' % c for c in coeffs]}")
        print(f"    Mean congruence: {np.mean(np.abs(coeffs)):.3f}")
        print(f"    (>.95 = excellent, >.85 = good, <.85 = poor factor recovery)")

    print(f"\n{'='*50}")
    print("SUMMARY TABLE")
    print(f"{'='*50}")
    summary = pd.DataFrame(results_table)
    print(summary.to_string(index=False))

    aucs = [r["csd_auc"] for r in results_table]
    kappas = [r["weighted_kappa"] for r in results_table]
    auc_range = max(aucs) - min(aucs)
    kappa_range = max(kappas) - min(kappas)

    print(f"\n  AUC range across K: {auc_range:.3f} {'(STABLE: <0.03)' if auc_range < 0.03 else '(MODERATE)' if auc_range < 0.05 else '(SENSITIVE)'}")
    print(f"  Kappa range across K: {kappa_range:.3f} {'(STABLE: <0.05)' if kappa_range < 0.05 else '(MODERATE)' if kappa_range < 0.10 else '(SENSITIVE)'}")

    summary.to_csv(os.path.join(OUTPUT_DIR, "k_sensitivity_results.csv"), index=False)
    print(f"\nSaved to robustness/k_sensitivity_results.csv")

    return summary


if __name__ == "__main__":
    run_k_sensitivity()
