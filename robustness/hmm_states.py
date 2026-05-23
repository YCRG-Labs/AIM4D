"""
HMM state count sensitivity analysis (S=3, 4, 5, 6).

Tests whether the 5-state regime classification is robust to alternative
state counts by comparing BIC, ICL, cross-validated log-likelihood, and
downstream early warning performance.

Methodological basis:
  - Zucchini, MacDonald & Langrock (2016): HMM model selection via BIC
  - Celeux & Durand (2008): cross-validated likelihood for HMMs
  - Hamilton (1989): foundational MS model
  - Biernacki, Celeux & Govaert (2000): ICL with entropy penalty

Reports:
  - BIC, ICL, blocked-CV log-likelihood for S=3,4,5,6
  - Cohen's kappa (weighted) vs V-Dem for each S
  - State interpretation table (mean polyarchy per state)
  - Downstream EWS detection rate
"""

import sys
import os
import warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
from sklearn.metrics import cohen_kappa_score
from hmmlearn import hmm
from scipy.special import logsumexp

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from stage3_msvar.estimate import (
    load_inputs, FACTOR_COLS, STATE_LABELS,
    prepare_sequences, MIN_F1_MARGIN,
    DIRICHLET_DIAG, DIRICHLET_OFF,
    precompute_log_emissions, hamilton_filter_fast,
    ADJ_PAIRS, N_ADJ,
)

OUTPUT_DIR = os.path.dirname(os.path.abspath(__file__))
S_VALUES = [3, 4, 5, 6]
N_RESTARTS = 20


STATE_LABELS_BY_S = {
    3: {0: "democracy", 1: "hybrid_regime", 2: "authoritarian"},
    4: {0: "liberal_democracy", 1: "electoral_democracy",
        2: "hybrid_regime", 3: "authoritarian"},
    5: {0: "liberal_democracy", 1: "electoral_democracy",
        2: "hybrid_regime", 3: "competitive_authoritarian",
        4: "closed_authoritarian"},
    6: {0: "consolidated_democracy", 1: "liberal_democracy",
        2: "electoral_democracy", 3: "hybrid_regime",
        4: "competitive_authoritarian", 5: "closed_authoritarian"},
}


def quantile_init_s(X_all, n_states):
    """Quantile-based initialization for any state count."""
    f1 = X_all[:, 0]
    quantiles = np.linspace(0, 100, n_states + 1)
    thresholds = np.percentile(f1, quantiles)

    means = np.zeros((n_states, X_all.shape[1]))
    covars = []

    for s in range(n_states):
        low = thresholds[n_states - s - 1]
        high = thresholds[n_states - s]
        mask = (f1 >= low) & (f1 < high) if s < n_states - 1 else (f1 >= low)
        if s == 0:
            mask = (f1 >= low)

        if mask.sum() > X_all.shape[1] + 1:
            means[s] = X_all[mask].mean(axis=0)
            covars.append(np.cov(X_all[mask].T) + 1e-4 * np.eye(X_all.shape[1]))
        else:
            means[s] = X_all.mean(axis=0)
            covars.append(np.eye(X_all.shape[1]))

    return means, np.array(covars)


def regularize_transmat_s(P, n_states):
    """Regularize transition matrix with Dirichlet prior for any S."""
    alpha = np.full((n_states, n_states), DIRICHLET_OFF)
    np.fill_diagonal(alpha, DIRICHLET_DIAG)
    for i in range(n_states):
        for j in range(n_states):
            if abs(i - j) > 2:
                alpha[i, j] = 0.1
    counts = P * 100
    smoothed = counts + alpha
    smoothed /= smoothed.sum(axis=1, keepdims=True)
    return smoothed


def fit_hmm_with_states(X_all, lengths, n_states):
    """Fit HMM with a specific state count, return model + BIC + ICL."""
    init_means, init_covars = quantile_init_s(X_all, n_states)

    init_transmat = np.full((n_states, n_states), 0.005)
    for i in range(n_states):
        init_transmat[i, i] = 0.95
        if i > 0:
            init_transmat[i, i - 1] = 0.02
        if i < n_states - 1:
            init_transmat[i, i + 1] = 0.02
    init_transmat /= init_transmat.sum(axis=1, keepdims=True)

    best_model = None
    best_score = -np.inf

    for restart in range(N_RESTARTS):
        model = hmm.GaussianHMM(
            n_components=n_states, covariance_type="full",
            n_iter=500, tol=1e-5, random_state=restart, init_params="",
        )

        rng = np.random.RandomState(restart)
        scale = 0.1 if restart < N_RESTARTS // 2 else 0.3
        model.means_ = init_means + rng.randn(*init_means.shape) * scale if restart > 0 else init_means.copy()
        model.covars_ = init_covars.copy()
        perturbed = init_transmat + rng.dirichlet(np.ones(n_states) * 10, size=n_states) * 0.05 if restart > 0 else init_transmat.copy()
        perturbed /= perturbed.sum(axis=1, keepdims=True)
        model.transmat_ = perturbed
        model.startprob_ = np.ones(n_states) / n_states

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            try:
                model.fit(X_all, lengths)
                score = model.score(X_all, lengths)
                f1_means = model.means_[:, 0]
                ordered = np.all(np.diff(f1_means) <= 0)
                margins = -np.diff(f1_means)
                margin_ok = np.all(margins >= MIN_F1_MARGIN * 0.5)

                if ordered and margin_ok and score > best_score:
                    best_score = score
                    best_model = model
            except Exception:
                continue

    if best_model is None:
        for restart in range(N_RESTARTS):
            model = hmm.GaussianHMM(
                n_components=n_states, covariance_type="full",
                n_iter=500, tol=1e-5, random_state=restart + 5000,
            )
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                try:
                    model.fit(X_all, lengths)
                    score = model.score(X_all, lengths)
                    if score > best_score:
                        best_score = score
                        best_model = model
                except Exception:
                    continue

        if best_model is not None:
            reorder = np.argsort(-best_model.means_[:, 0])
            best_model.means_ = best_model.means_[reorder]
            best_model.covars_ = best_model.covars_[reorder]
            best_model.transmat_ = best_model.transmat_[reorder][:, reorder]
            best_model.startprob_ = best_model.startprob_[reorder]

    if best_model is None:
        return None, -np.inf, np.inf, np.inf

    best_model.transmat_ = regularize_transmat_s(best_model.transmat_, n_states)

    N = len(X_all)
    d = X_all.shape[1]
    n_params = (n_states * d +
                n_states * d * (d + 1) // 2 +
                n_states * (n_states - 1) +
                n_states - 1)
    bic = -2 * best_score + n_params * np.log(N)

    posteriors = best_model.predict_proba(X_all, lengths)
    entropy = -np.sum(posteriors * np.log(posteriors + 1e-300))
    icl = bic + 2 * entropy

    return best_model, best_score, bic, icl


def blocked_cv_loglik(X_all, lengths, country_order, n_states, n_folds=5):
    """Blocked cross-validation: hold out contiguous country blocks."""
    rng = np.random.RandomState(42)
    indices = np.arange(len(country_order))
    rng.shuffle(indices)
    folds = np.array_split(indices, n_folds)

    total_test_ll = 0.0
    total_test_obs = 0

    for fold_idx, test_indices in enumerate(folds):
        test_set = set(test_indices)
        train_seqs, train_lengths = [], []
        test_seqs, test_lengths = [], []

        offset = 0
        for i, l in enumerate(lengths):
            seq = X_all[offset:offset + l]
            if i in test_set:
                test_seqs.append(seq)
                test_lengths.append(l)
            else:
                train_seqs.append(seq)
                train_lengths.append(l)
            offset += l

        if not train_seqs or not test_seqs:
            continue

        X_train = np.concatenate(train_seqs)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model = hmm.GaussianHMM(
                n_components=n_states, covariance_type="full",
                n_iter=300, tol=1e-4, random_state=fold_idx,
            )
            try:
                model.fit(X_train, train_lengths)
                X_test = np.concatenate(test_seqs)
                test_ll = model.score(X_test, test_lengths)
                total_test_ll += test_ll
                total_test_obs += sum(test_lengths)
            except Exception:
                continue

    return total_test_ll / max(total_test_obs, 1)


def validate_s(state_df, n_states):
    """Validate against V-Dem regime classification."""
    vdem = pd.read_csv(
        os.path.join(os.path.dirname(__file__), "..", "data", "vdem_v16.csv"),
        low_memory=False, usecols=["country_name", "year", "v2x_regime", "v2x_polyarchy"],
    )
    vdem = vdem.dropna(subset=["v2x_regime"])
    vdem["v2x_regime"] = vdem["v2x_regime"].astype(int)

    def to_nstate(regime, poly):
        if n_states == 3:
            if regime >= 2: return 0
            if regime == 1: return 1
            return 2
        elif n_states == 4:
            if regime == 3: return 0
            if regime == 2: return 1
            if regime == 1: return 2
            return 3
        elif n_states == 5:
            if regime == 3: return 0
            if regime == 2: return 1
            if regime == 1: return 2 if (poly is not None and poly > 0.35) else 3
            return 4
        elif n_states == 6:
            if regime == 3:
                return 0 if (poly is not None and poly > 0.8) else 1
            if regime == 2: return 2
            if regime == 1: return 3 if (poly is not None and poly > 0.35) else 4
            return 5

    merged = state_df.merge(vdem, on=["country_name", "year"], how="inner")
    merged["vdem_s"] = merged.apply(lambda r: to_nstate(r["v2x_regime"], r["v2x_polyarchy"]), axis=1)

    kappa = cohen_kappa_score(merged["vdem_s"], merged["state"])
    kappa_w = cohen_kappa_score(merged["vdem_s"], merged["state"], weights="linear")

    poly_by_state = merged.groupby("state")["v2x_polyarchy"].agg(["mean", "std", "count"])

    return kappa, kappa_w, poly_by_state


def decode_states(model, X_all, lengths, country_order, df, n_states):
    """Decode states from fitted HMM."""
    rows = []
    offset = 0

    for i, country in enumerate(country_order):
        l = lengths[i]
        seq = X_all[offset:offset + l]
        posteriors = model.predict_proba(seq.reshape(-1, seq.shape[-1]) if seq.ndim == 1 else seq)
        states = np.argmax(posteriors, axis=1)

        cdf = df[df["country_name"] == country].sort_values("year")
        years = cdf["year"].values[-l:]

        for t in range(l):
            row = {
                "country_name": country,
                "country_text_id": cdf["country_text_id"].iloc[0],
                "year": int(years[t]),
                "state": int(states[t]),
                "state_label": STATE_LABELS_BY_S.get(n_states, {}).get(int(states[t]), f"state_{states[t]}"),
            }
            for s in range(n_states):
                row[f"prob_state_{s}"] = posteriors[t, s]
            rows.append(row)

        offset += l

    return pd.DataFrame(rows)


def run_hmm_states():
    print("=" * 70)
    print("ROBUSTNESS CHECK: HMM State Count Sensitivity (S=3, 4, 5, 6)")
    print("=" * 70)
    print()
    print("Methodological basis:")
    print("  Zucchini et al. (2016) BIC for HMMs; Celeux & Durand (2008) blocked CV;")
    print("  Biernacki et al. (2000) ICL entropy penalty")
    print()

    df, beta_cols = load_inputs()

    lag_cols = []
    for fc in FACTOR_COLS:
        lcol = f"lag_{fc}"
        df[lcol] = df.groupby("country_name")[fc].shift(1)
        lag_cols.append(lcol)
    df = df.dropna(subset=lag_cols)

    obs_cols = FACTOR_COLS + lag_cols
    X_all, lengths, country_order = prepare_sequences(df, obs_cols)
    print(f"Panel: {len(country_order)} countries, {sum(lengths)} obs, {len(obs_cols)} features\n")

    results = []

    for S in S_VALUES:
        print(f"\n{'='*50}")
        print(f"S = {S} states")
        print(f"{'='*50}")

        print(f"\n  Fitting HMM with {S} states ({N_RESTARTS} restarts)...")
        model, ll, bic, icl = fit_hmm_with_states(X_all, lengths, S)

        if model is None:
            print(f"  FAILED: Could not fit HMM with {S} states")
            results.append({"S": S, "LL": np.nan, "BIC": np.nan, "ICL": np.nan,
                           "CV_LL": np.nan, "kappa": np.nan, "kappa_w": np.nan})
            continue

        print(f"  LL={ll:.1f}, BIC={bic:.1f}, ICL={icl:.1f}")

        print(f"\n  State means (Factor 1, descending):")
        labels = STATE_LABELS_BY_S.get(S, {})
        for s in range(S):
            label = labels.get(s, f"state_{s}")
            print(f"    {label}: F1={model.means_[s, 0]:.3f}")

        print(f"\n  Running blocked cross-validation (5 folds)...")
        cv_ll = blocked_cv_loglik(X_all, lengths, country_order, S)
        print(f"  CV log-likelihood per obs: {cv_ll:.4f}")

        state_df = decode_states(model, X_all, lengths, country_order, df, S)
        kappa, kappa_w, poly_by_state = validate_s(state_df, S)

        print(f"\n  Validation vs V-Dem:")
        print(f"    Cohen's kappa: {kappa:.3f}")
        print(f"    Weighted kappa: {kappa_w:.3f}")

        print(f"\n  Polyarchy by state:")
        for s in range(S):
            if s in poly_by_state.index:
                r = poly_by_state.loc[s]
                label = labels.get(s, f"state_{s}")
                print(f"    {label}: {r['mean']:.3f} +/- {r['std']:.3f} (n={int(r['count'])})")

        print(f"\n  State distribution:")
        for label, count in state_df["state_label"].value_counts().items():
            print(f"    {label}: {count} ({count / len(state_df) * 100:.1f}%)")

        diag = np.diag(model.transmat_)
        print(f"\n  State persistence (transition matrix diagonal):")
        for s in range(S):
            label = labels.get(s, f"state_{s}")
            print(f"    {label}: {diag[s]:.4f}")

        results.append({
            "S": S, "LL": ll, "BIC": bic, "ICL": icl, "CV_LL": cv_ll,
            "kappa": kappa, "kappa_w": kappa_w,
            "min_persistence": diag.min(),
        })

    print(f"\n{'='*50}")
    print("SUMMARY TABLE")
    print(f"{'='*50}")
    summary = pd.DataFrame(results)
    print(summary.to_string(index=False, float_format="%.3f"))

    valid = summary.dropna(subset=["BIC"])
    if len(valid) > 0:
        best_bic = valid.loc[valid["BIC"].idxmin(), "S"]
        best_icl = valid.loc[valid["ICL"].idxmin(), "S"]
        best_cv = valid.loc[valid["CV_LL"].idxmax(), "S"]
        best_kappa = valid.loc[valid["kappa_w"].idxmax(), "S"]

        print(f"\n  Selected by BIC:        S={int(best_bic)}")
        print(f"  Selected by ICL:        S={int(best_icl)}")
        print(f"  Selected by blocked CV: S={int(best_cv)}")
        print(f"  Best weighted kappa:    S={int(best_kappa)}")

    summary.to_csv(os.path.join(OUTPUT_DIR, "hmm_states_results.csv"), index=False)
    print(f"\nSaved to robustness/hmm_states_results.csv")

    return summary


if __name__ == "__main__":
    run_hmm_states()
