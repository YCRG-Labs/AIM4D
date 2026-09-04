"""
HMM state count sensitivity under the LOCKED Stage 3 specification.

hmm_states.py sweeps S under a specification that differs from the pipeline's
own in four ways: it uses a full rather than diagonal covariance, 20 rather
than 60 restarts, no posterior stabilization filter, and it fits on the whole
panel rather than the pre-cutoff training subset. Those choices are why its
S=5 weighted kappa reads 0.67 against the locked pipeline's 0.72, a gap that
the paper previously reported without reconciling.

This script re-runs the same sweep under the locked specification: diagonal
covariance with min_covar 0.05, Dirichlet-regularized transitions, 60
restarts with the ordering and separation constraints, a training fit
restricted to year <= AIM4D_CUTOFF, and decoding through the duration
dependent stabilization filter. Agreement is scored with the same
state-count-specific V-Dem mapping hmm_states.py uses, so the columns stay
comparable across S.

DURATION_PARAMS is keyed by state index and covers five states; for S=6 the
sixth index falls through to a zero duration bonus, and for S<5 the surplus
keys are unused.

Outputs robustness/hmm_states_locked_results.csv.
"""

import os
import sys
import warnings

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from hmmlearn import hmm
from sklearn.metrics import cohen_kappa_score

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import stage3_msvar.estimate as est
from stage3_msvar.estimate import (
    load_inputs, FACTOR_COLS, prepare_sequences, quantile_init,
    transmat_alpha, regularize_transmat, precompute_log_emissions,
    hamilton_filter_fast, stabilize_states, MIN_F1_MARGIN, N_ADJ,
)

for _s in range(5, max(6, 8)):
    est.STATE_LABELS.setdefault(_s, f"state_{_s}")

OUTPUT_DIR = os.path.dirname(os.path.abspath(__file__))
S_VALUES = [3, 4, 5, 6]
N_RESTARTS = int(os.environ.get("AIM4D_HMM_RESTARTS", "60"))
MAX_TRAIN_YEAR = int(os.environ.get("AIM4D_CUTOFF", "2019"))


def fit_locked(X_train, lengths_train, n_states):
    init_means, init_covars = quantile_init(X_train, n_states=n_states)
    init_covars_diag = np.maximum(np.array([np.diag(c) for c in init_covars]), 0.05)

    init_transmat = np.full((n_states, n_states), 0.005)
    for i in range(n_states):
        init_transmat[i, i] = 0.95
        if i > 0:
            init_transmat[i, i - 1] = 0.02
        if i < n_states - 1:
            init_transmat[i, i + 1] = 0.02
    init_transmat /= init_transmat.sum(axis=1, keepdims=True)

    alpha = transmat_alpha(n_states=n_states)

    best_model, best_score = None, -np.inf
    for restart in range(N_RESTARTS):
        model = hmm.GaussianHMM(
            n_components=n_states, covariance_type="diag", min_covar=0.05,
            transmat_prior=alpha, implementation="log",
            n_iter=500, tol=1e-5, random_state=restart, init_params="",
        )
        rng = np.random.RandomState(restart)
        scale = 0.1 if restart < N_RESTARTS // 2 else 0.3
        model.means_ = (init_means + rng.randn(*init_means.shape) * scale
                        if restart > 0 else init_means.copy())
        model.covars_ = init_covars_diag.copy()
        perturbed = (init_transmat + rng.dirichlet(np.ones(n_states) * 10, size=n_states) * 0.05
                     if restart > 0 else init_transmat.copy())
        perturbed /= perturbed.sum(axis=1, keepdims=True)
        model.transmat_ = perturbed
        model.startprob_ = np.ones(n_states) / n_states

        try:
            model.fit(X_train, lengths_train)
            score = model.score(X_train, lengths_train)
            f1_means = model.means_[:, 0]
            if np.all(np.diff(f1_means) <= 0) and np.all(-np.diff(f1_means) >= MIN_F1_MARGIN):
                if score > best_score:
                    best_score, best_model = score, model
        except Exception:
            continue

    if best_model is None:
        return None, np.nan, np.nan

    best_model.transmat_ = regularize_transmat(best_model.transmat_, n_states=n_states)

    n_obs, n_dim = X_train.shape
    n_params = (n_states - 1) + n_states * (n_states - 1) + 2 * n_states * n_dim
    bic = -2 * best_score + n_params * np.log(n_obs)

    return best_model, best_score, bic


def decode_locked(model, X_all, lengths, country_order, df, n_states):
    log_emit_all = precompute_log_emissions(X_all, model.means_, model.covars_)
    rows = []
    idx = 0
    for i, country in enumerate(country_order):
        L = lengths[i]
        posteriors, states, _ = hamilton_filter_fast(
            log_emit_all[idx:idx + L], model.startprob_, model.transmat_,
            np.zeros((N_ADJ, 1)), None,
        )
        states = stabilize_states(states, posteriors)
        cdf = df[df["country_name"] == country].sort_values("year")
        years = cdf["year"].values[-len(states):]
        for t in range(len(states)):
            rows.append({"country_name": country, "year": int(years[t]),
                         "state": int(states[t])})
        idx += L
    return pd.DataFrame(rows)


def validate_s(state_df, n_states):
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
    return (cohen_kappa_score(merged["vdem_s"], merged["state"]),
            cohen_kappa_score(merged["vdem_s"], merged["state"], weights="linear"),
            len(merged))


def main():
    print("=== HMM state sweep under the LOCKED Stage 3 specification ===\n")
    print(f"restarts={N_RESTARTS}  covariance=diag  train cutoff={MAX_TRAIN_YEAR}  "
          f"stabilization=on\n")

    df, beta_cols = load_inputs()
    lag_cols = []
    for fc in FACTOR_COLS:
        lcol = f"lag_{fc}"
        df[lcol] = df.groupby("country_name")[fc].shift(1)
        lag_cols.append(lcol)
    df = df.dropna(subset=lag_cols)
    obs_cols = FACTOR_COLS + lag_cols

    X_all, lengths, country_order = prepare_sequences(df, obs_cols)
    df_train = df[df["year"] <= MAX_TRAIN_YEAR].copy()
    X_train, lengths_train, _ = prepare_sequences(df_train, obs_cols)
    print(f"full panel: {len(country_order)} countries, {sum(lengths)} obs")
    print(f"train subset: {sum(lengths_train)} obs\n")

    rows = []
    for S in S_VALUES:
        print(f"--- S = {S} ---")
        model, ll, bic = fit_locked(X_train, lengths_train, S)
        if model is None:
            print("  no valid model under the ordering and separation constraints\n")
            rows.append({"S": S, "LL": np.nan, "BIC": np.nan,
                         "kappa": np.nan, "kappa_w": np.nan, "n_matched": np.nan,
                         "min_persistence": np.nan})
            continue
        state_df = decode_locked(model, X_all, lengths, country_order, df, S)
        kappa, kappa_w, n_matched = validate_s(state_df, S)
        min_persist = float(np.min(np.diag(model.transmat_)))
        print(f"  LL={ll:.1f}  BIC={bic:.1f}  kappa={kappa:.3f}  "
              f"kappa_w={kappa_w:.3f}  n={n_matched}  min_persist={min_persist:.3f}\n")
        rows.append({"S": S, "LL": ll, "BIC": bic, "kappa": kappa,
                     "kappa_w": kappa_w, "n_matched": n_matched,
                     "min_persistence": min_persist})

    out = pd.DataFrame(rows)
    out_path = os.path.join(OUTPUT_DIR, "hmm_states_locked_results.csv")
    out.to_csv(out_path, index=False)
    print(out.to_string(index=False))
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
