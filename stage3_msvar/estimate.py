import sys
import os
import warnings

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "BLIS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, "4")

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, cohen_kappa_score
from sklearn.linear_model import LogisticRegressionCV
from scipy.optimize import minimize
from scipy.special import softmax, logsumexp
from hmmlearn import hmm
from joblib import Parallel, delayed

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

FACTOR_COLS = ["factor_1", "factor_2", "factor_3", "factor_4"]
N_STATES = 5
N_RESTARTS = int(os.environ.get("AIM4D_HMM_RESTARTS",
                                 "10" if os.environ.get("AIM4D_QUICK") == "1" else "60"))
DIRICHLET_DIAG = 50
DIRICHLET_OFF = 2
MAX_TRAIN_YEAR = int(os.environ.get("AIM4D_CUTOFF", "2019"))
EXCLUDE_COUNTRY = os.environ.get("AIM4D_EXCLUDE_COUNTRY", "").strip() or None
MIN_F1_MARGIN = 0.15
STATE_LABELS = {
    0: "liberal_democracy",
    1: "electoral_democracy",
    2: "hybrid_regime",
    3: "competitive_authoritarian",
    4: "closed_authoritarian",
}


def load_inputs():
    factors = pd.read_csv(
        os.path.join(os.path.dirname(__file__), "..", "stage1_factors", "country_year_factors.csv")
    )
    betas = pd.read_csv(
        os.path.join(os.path.dirname(__file__), "..", "stage2_betas", "country_year_betas.csv")
    )
    beta_cols = [c for c in betas.columns if c.startswith("beta_")]
    merged = factors.merge(betas[["country_name", "year"] + beta_cols], on=["country_name", "year"])
    return merged, beta_cols


def load_macro():
    path = os.path.join(os.path.dirname(__file__), "..", "data", "macro_covariates.csv")
    if not os.path.exists(path):
        try:
            import wbgapi as wb
            indicators = {
                "NY.GDP.PCAP.KD": "gdp_pc",
                "NY.GDP.MKTP.KD.ZG": "gdp_growth",
                "SP.URB.TOTL.IN.ZS": "urbanization",
                "NY.GDP.TOTL.RT.ZS": "resource_rents",
                "NE.TRD.GNFS.ZS": "trade_openness",
                "MS.MIL.XPND.GD.ZS": "military_spending",
            }
            frames = []
            for code, name in indicators.items():
                try:
                    raw = wb.data.DataFrame(code, time=range(1970, 2026), labels=False)
                    long = raw.stack().reset_index()
                    long.columns = ["iso3", "year", name]
                    long["year"] = long["year"].astype(str).str.replace("YR", "").astype(int)
                    frames.append(long)
                except Exception:
                    pass
            if frames:
                macro = frames[0]
                for f in frames[1:]:
                    macro = macro.merge(f, on=["iso3", "year"], how="outer")
                macro.to_csv(path, index=False)
                return macro
        except ImportError:
            pass
        return None
    return pd.read_csv(path)


def prepare_sequences(df, obs_cols):
    countries = sorted(df["country_name"].unique())
    sequences = []
    lengths = []
    country_order = []

    for country in countries:
        cdf = df[df["country_name"] == country].sort_values("year")
        if len(cdf) < 10:
            continue
        X = cdf[obs_cols].values
        if np.any(np.isnan(X)):
            continue
        sequences.append(X)
        lengths.append(len(X))
        country_order.append(country)

    X_all = np.concatenate(sequences)
    return X_all, lengths, country_order


def quantile_init(X_all, n_states=N_STATES):
    f1 = X_all[:, 0]
    quantiles = np.linspace(0, 100, n_states + 1)
    thresholds = np.percentile(f1, quantiles)

    means = np.zeros((n_states, X_all.shape[1]))
    covars = []

    print(f"Quantile-based init (Factor 1 bands, no V-Dem supervision):")
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

        print(f"  Band {s} ({STATE_LABELS[s]}): {mask.sum()} obs, "
              f"F1 range=[{low:.3f}, {high:.3f}], F1 mean={means[s, 0]:.3f}")

    return means, np.array(covars)


def transmat_alpha(n_states=N_STATES):
    """
    Asymmetric Dirichlet alpha matrix for the transition prior (Svolik 2008,
    Epstein et al. 2006). Passed to hmmlearn's `transmat_prior` so the prior
    actually constrains EM, not just the final transmat.
    """
    diag_priors = {
        0: 500,
        1: 200,
        2: DIRICHLET_DIAG,
        3: DIRICHLET_DIAG,
        4: 200,
    }
    alpha = np.full((n_states, n_states), DIRICHLET_OFF, dtype=float)
    for i in range(n_states):
        alpha[i, i] = diag_priors.get(i, DIRICHLET_DIAG)
        for j in range(n_states):
            if abs(i - j) > 2:
                alpha[i, j] = 0.1
    return alpha


def regularize_transmat(P, n_states=N_STATES):
    """Post-hoc smoothing of fitted transmat with the same Dirichlet prior."""
    alpha = transmat_alpha(n_states)
    counts = P * 100
    smoothed = counts + alpha
    smoothed /= smoothed.sum(axis=1, keepdims=True)
    return smoothed


def _fit_one_restart(restart, X_all, lengths, init_means, init_covars_diag,
                     init_transmat, alpha):
    model = hmm.GaussianHMM(
        n_components=N_STATES,
        covariance_type="diag",
        min_covar=0.05,
        transmat_prior=alpha,
        implementation="log",
        n_iter=500,
        tol=1e-5,
        random_state=restart,
        init_params="",
    )

    rng = np.random.RandomState(restart)
    scale = 0.1 if restart < N_RESTARTS // 2 else 0.3
    model.means_ = init_means + rng.randn(*init_means.shape) * scale if restart > 0 else init_means.copy()
    model.covars_ = init_covars_diag.copy()
    perturbed = init_transmat + rng.dirichlet(np.ones(N_STATES) * 10, size=N_STATES) * 0.05 if restart > 0 else init_transmat.copy()
    perturbed /= perturbed.sum(axis=1, keepdims=True)
    model.transmat_ = perturbed
    model.startprob_ = np.ones(N_STATES) / N_STATES

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            model.fit(X_all, lengths)
            score = model.score(X_all, lengths)
            f1_means = model.means_[:, 0]
            ordered = np.all(np.diff(f1_means) <= 0)
            margins = -np.diff(f1_means)
            well_separated = np.all(margins >= MIN_F1_MARGIN)
            if ordered and well_separated:
                return model, score
        except Exception:
            pass
    return None


def fit_baseline_hmm(X_all, lengths, init_means, init_covars):
    init_transmat = np.full((N_STATES, N_STATES), 0.005)
    for i in range(N_STATES):
        init_transmat[i, i] = 0.95
        if i > 0:
            init_transmat[i, i - 1] = 0.02
        if i < N_STATES - 1:
            init_transmat[i, i + 1] = 0.02
    init_transmat /= init_transmat.sum(axis=1, keepdims=True)

    init_covars_diag = np.array([np.diag(c) for c in init_covars])
    init_covars_diag = np.maximum(init_covars_diag, 0.05)
    alpha = transmat_alpha()

    best_model = None
    best_score = -np.inf

    n_jobs = int(os.environ.get("AIM4D_THREADS", "4"))
    results = Parallel(n_jobs=n_jobs, backend="loky")(
        delayed(_fit_one_restart)(
            restart, X_all, lengths, init_means, init_covars_diag,
            init_transmat, alpha,
        )
        for restart in range(N_RESTARTS)
    )

    for res in results:
        if res is None:
            continue
        model, score = res
        if score > best_score:
            best_score = score
            best_model = model

    status = f"score={best_score:.1f}" if best_model else "no valid model"
    print(f"  {N_RESTARTS} restarts complete: {status}")

    if best_model is None:
        print("  Fallback: unconstrained + reorder")
        for restart in range(N_RESTARTS):
            model = hmm.GaussianHMM(
                n_components=N_STATES, covariance_type="diag",
                min_covar=0.05, transmat_prior=alpha, implementation="log",
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
        reorder = np.argsort(-best_model.means_[:, 0])
        best_model.means_ = best_model.means_[reorder]
        best_model.covars_ = best_model.covars_[reorder]
        best_model.transmat_ = best_model.transmat_[reorder][:, reorder]
        best_model.startprob_ = best_model.startprob_[reorder]

    best_model.transmat_ = regularize_transmat(best_model.transmat_)
    print(f"\nBaseline HMM: LL={best_score:.1f}, min persistence={np.min(np.diag(best_model.transmat_)):.3f}")
    return best_model, best_score


def precompute_log_emissions(X_all, means, covars):
    K = len(means)
    N = len(X_all)
    d = X_all.shape[1]
    log_emit = np.zeros((N, K))

    is_diag = covars.ndim == 2

    for k in range(K):
        diff = X_all - means[k]
        if is_diag:
            var = np.maximum(covars[k], 1e-6)
            logdet = np.sum(np.log(var))
            mahal = np.sum((diff ** 2) / var, axis=1)
        else:
            sign, logdet = np.linalg.slogdet(covars[k])
            cov_inv = np.linalg.inv(covars[k])
            mahal = np.sum(diff @ cov_inv * diff, axis=1)
        log_emit[:, k] = -0.5 * (d * np.log(2 * np.pi) + logdet + mahal)

    return log_emit


ADJ_PAIRS = [(0, 1), (1, 0), (1, 2), (2, 1), (2, 3), (3, 2), (3, 4), (4, 3)]
N_ADJ = len(ADJ_PAIRS)


def tvtp_transmat(z_t, theta, base_transmat):
    K = base_transmat.shape[0]
    log_P = np.log(base_transmat + 1e-300).copy()

    for idx, (i, j) in enumerate(ADJ_PAIRS):
        log_P[i, j] += z_t @ theta[idx]

    for i in range(K):
        log_P[i] -= logsumexp(log_P[i])

    return np.exp(log_P)


def hamilton_filter_fast(log_emit_seq, startprob, base_transmat, theta, Z_seq):
    T, K = log_emit_seq.shape
    log_alpha = np.zeros((T, K))
    log_alpha[0] = np.log(startprob + 1e-300) + log_emit_seq[0]

    has_covs = Z_seq is not None

    for t in range(1, T):
        if has_covs:
            P_t = tvtp_transmat(Z_seq[t], theta, base_transmat)
        else:
            P_t = base_transmat
        log_P = np.log(P_t + 1e-300)

        for j in range(K):
            log_alpha[t, j] = logsumexp(log_alpha[t - 1] + log_P[:, j]) + log_emit_seq[t, j]

    ll = logsumexp(log_alpha[-1])

    log_beta = np.zeros((T, K))
    for t in range(T - 2, -1, -1):
        if has_covs:
            P_t = tvtp_transmat(Z_seq[t + 1], theta, base_transmat)
        else:
            P_t = base_transmat
        log_P = np.log(P_t + 1e-300)
        for j in range(K):
            log_beta[t, j] = logsumexp(log_P[j, :] + log_emit_seq[t + 1] + log_beta[t + 1])

    log_gamma = log_alpha + log_beta
    for t in range(T):
        log_gamma[t] -= logsumexp(log_gamma[t])

    posteriors = np.exp(log_gamma)
    states = np.argmax(posteriors, axis=1)
    return posteriors, states, ll


def tvtp_neg_loglik_fast(params, emit_seqs, Z_seqs, startprob, base_transmat, n_covs):
    theta = params.reshape(N_ADJ, n_covs)
    total_ll = 0.0
    for le, zs in zip(emit_seqs, Z_seqs):
        _, _, ll = hamilton_filter_fast(le, startprob, base_transmat, theta, zs)
        total_ll += ll
    return -total_ll


def fit_tvtp(emit_seqs, Z_seqs, baseline_model, n_covs):
    startprob = baseline_model.startprob_
    base_transmat = baseline_model.transmat_

    n_params = N_ADJ * n_covs
    theta_init = np.zeros(n_params)

    print(f"  TVTP: {N_ADJ} adjacent transitions x {n_covs} covariates = {n_params} params")

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result = minimize(
            tvtp_neg_loglik_fast, theta_init,
            args=(emit_seqs, Z_seqs, startprob, base_transmat, n_covs),
            method="L-BFGS-B",
            bounds=[(-3, 3)] * n_params,
            options={"maxiter": 100, "ftol": 1e-4},
        )

    theta = result.x.reshape(N_ADJ, n_covs)
    print(f"  TVTP converged={result.success}, LL={-result.fun:.1f}, iters={result.nit}")
    return theta


POSTERIOR_THRESHOLD = 0.65
DURATION_PARAMS = {
    0: (15.0, 0.20),
    1: (10.0, 0.15),
    2: (5.0, 0.10),
    3: (5.0, 0.10),
    4: (8.0, 0.15),
}


def stabilize_states(states, posteriors):
    """
    Posterior threshold + duration-dependent stabilization.
    Requires supermajority posterior (0.65+) to reclassify, with the
    threshold increasing for long-duration states (Svolik 2008).
    Eliminates boundary oscillations (Canada, Portugal, South Korea)
    while preserving genuine transitions (Hungary 2010).
    """
    T = len(states)
    stabilized = states.copy()
    duration = 1

    for t in range(1, T):
        current_state = stabilized[t - 1]
        proposed_state = states[t]

        if proposed_state != current_state:
            base = POSTERIOR_THRESHOLD
            if current_state in DURATION_PARAMS:
                half_life, max_bonus = DURATION_PARAMS[current_state]
                bonus = max_bonus * (1.0 - np.exp(-duration / half_life)) if duration > 0 else 0
            else:
                bonus = 0
            threshold = min(0.90, base + bonus)

            if posteriors[t, proposed_state] >= threshold:
                stabilized[t] = proposed_state
                duration = 1
            else:
                stabilized[t] = current_state
                duration += 1
        else:
            duration += 1

    return stabilized


def decode_all(emit_seqs, Z_seqs, lengths, country_order, df, baseline_model, theta=None):
    rows = []
    total_ll = 0.0

    for i, country in enumerate(country_order):
        posteriors, states, ll = hamilton_filter_fast(
            emit_seqs[i], baseline_model.startprob_, baseline_model.transmat_,
            theta if theta is not None else np.zeros((N_ADJ, 1)),
            Z_seqs[i] if Z_seqs else None,
        )
        total_ll += ll

        states = stabilize_states(states, posteriors)

        cdf = df[df["country_name"] == country].sort_values("year")
        years = cdf["year"].values[-len(states):]

        for t in range(len(states)):
            row = {
                "country_name": country,
                "country_text_id": cdf["country_text_id"].iloc[0],
                "year": int(years[t]),
                "state": int(states[t]),
                "state_label": STATE_LABELS.get(int(states[t]), f"state_{states[t]}"),
            }
            for s in range(N_STATES):
                row[f"prob_state_{s}"] = posteriors[t, s]
            rows.append(row)

    return pd.DataFrame(rows), total_ll


def validate(state_df):
    vdem = pd.read_csv(
        os.path.join(os.path.dirname(__file__), "..", "data", "vdem_v16.csv"),
        low_memory=False, usecols=["country_name", "year", "v2x_regime", "v2x_polyarchy"],
    )
    vdem = vdem.dropna(subset=["v2x_regime"])
    vdem["v2x_regime"] = vdem["v2x_regime"].astype(int)

    def to_5state(regime, poly):
        if regime == 3: return 0
        if regime == 2: return 1
        if regime == 1: return 2 if (poly is not None and poly > 0.35) else 3
        return 4

    merged = state_df.merge(vdem, on=["country_name", "year"], how="inner")
    merged["vdem_5"] = merged.apply(lambda r: to_5state(r["v2x_regime"], r["v2x_polyarchy"]), axis=1)

    our = merged["state"].values
    ref = merged["vdem_5"].values

    kappa = cohen_kappa_score(ref, our)
    kappa_w = cohen_kappa_score(ref, our, weights="linear")
    accuracy = np.mean(ref == our)

    print(f"\n=== Validation vs V-Dem (5-state) ===")
    print(f"Cohen's kappa: {kappa:.3f}")
    print(f"Weighted kappa (linear): {kappa_w:.3f}")
    print(f"Accuracy: {accuracy:.3f}")

    labels = list(range(N_STATES))
    cm = confusion_matrix(ref, our, labels=labels)
    names = ["lib_dem", "elec_dem", "hybrid", "comp_auth", "clos_auth"]
    print(f"\nConfusion matrix (rows=V-Dem, cols=ours):")
    print(pd.DataFrame(cm, index=names, columns=names))

    print(f"\nPolyarchy by our state:")
    by_state = merged.groupby("state")["v2x_polyarchy"].agg(["mean", "std", "count"])
    for s in range(N_STATES):
        if s in by_state.index:
            r = by_state.loc[s]
            print(f"  {STATE_LABELS[s]}: {r['mean']:.3f} ± {r['std']:.3f} (n={int(r['count'])})")

    return kappa, kappa_w


def lasso_select(state_df, macro, covariate_cols):
    merged = state_df.merge(macro, left_on=["country_text_id", "year"], right_on=["iso3", "year"], how="inner")
    merged = merged.dropna(subset=covariate_cols + ["state"])

    scaler = StandardScaler()
    X = scaler.fit_transform(merged[covariate_cols])
    y = merged["state"].values

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        lasso = LogisticRegressionCV(
            penalty="l1", solver="saga", cv=5, Cs=20,
            multi_class="multinomial", max_iter=5000, random_state=42,
            n_jobs=int(os.environ.get("AIM4D_THREADS", "4")),
        )
        lasso.fit(X, y)

    coef_norms = np.abs(lasso.coef_).sum(axis=0)
    selected = [(covariate_cols[i], coef_norms[i]) for i in range(len(covariate_cols)) if coef_norms[i] > 0.01]
    selected.sort(key=lambda x: -x[1])
    print(f"\nLASSO: {len(selected)}/{len(covariate_cols)} covariates selected:")
    for name, imp in selected:
        print(f"  {name}: {imp:.4f}")
    return [s[0] for s in selected]


def run_stage3():
    print("=== Stage 3: MS Regime Classification + TVTP ===\n")

    df, beta_cols = load_inputs()

    lag_cols = []
    for fc in FACTOR_COLS:
        lcol = f"lag_{fc}"
        df[lcol] = df.groupby("country_name")[fc].shift(1)
        lag_cols.append(lcol)
    df = df.dropna(subset=lag_cols)

    obs_cols = FACTOR_COLS + lag_cols
    print(f"Features ({len(obs_cols)}): {FACTOR_COLS} + {lag_cols} (MS-VAR(1) with lagged factors)")

    X_all, lengths, country_order = prepare_sequences(df, obs_cols)
    print(f"Panel: {len(country_order)} countries, {sum(lengths)} obs\n")

    df_train = df[df["year"] <= MAX_TRAIN_YEAR].copy()
    if EXCLUDE_COUNTRY:
        df_train = df_train[df_train["country_name"] != EXCLUDE_COUNTRY]
        print(f"  Task F: excluding country '{EXCLUDE_COUNTRY}' from HMM/TVTP fit")
    X_train, lengths_train, country_order_train = prepare_sequences(df_train, obs_cols)
    print(f"Training subset: year <= {MAX_TRAIN_YEAR}, "
          f"{len(country_order_train)} countries, {sum(lengths_train)} obs\n")

    init_means, init_covars = quantile_init(X_train)

    print(f"\nPhase 1: Baseline HMM (K-means init, no supervision)...")
    baseline, base_score = fit_baseline_hmm(X_train, lengths_train, init_means, init_covars)

    print(f"\nState means:")
    for s in range(N_STATES):
        m = ", ".join(f"{baseline.means_[s, i]:.3f}" for i in range(len(obs_cols)))
        print(f"  {STATE_LABELS[s]}: [{m}]")
    print(f"\nBaseline transition matrix:")
    for i in range(N_STATES):
        row = " ".join(f"{baseline.transmat_[i,j]:.4f}" for j in range(N_STATES))
        print(f"  {STATE_LABELS[i]:30s} [{row}]")

    log_emit_all = precompute_log_emissions(X_all, baseline.means_, baseline.covars_)

    emit_seqs = []
    obs_seqs = []
    idx = 0
    for l in lengths:
        emit_seqs.append(log_emit_all[idx:idx + l])
        obs_seqs.append(X_all[idx:idx + l])
        idx += l

    baseline_state_df, _ = decode_all(
        emit_seqs, [None] * len(emit_seqs), lengths, country_order, df, baseline,
    )
    print(f"\nBaseline state distribution:")
    for label, count in baseline_state_df["state_label"].value_counts().items():
        print(f"  {label}: {count} ({count / len(baseline_state_df) * 100:.1f}%)")
    k_base, kw_base = validate(baseline_state_df)

    print(f"\n{'='*60}")
    print(f"Phase 2: TVTP with macro covariates\n")

    macro = load_macro()
    gdelt_path = os.path.join(os.path.dirname(__file__), "..", "data", "gdelt_country_year.csv")
    if os.path.exists(gdelt_path):
        gdelt = pd.read_csv(gdelt_path)
        gdelt = gdelt.rename(columns={"country_code": "iso3"})
        if macro is not None:
            macro = macro.merge(gdelt, on=["iso3", "year"], how="outer")
        else:
            macro = gdelt
        print(f"  GDELT loaded: {len(gdelt)} rows, covariates: {[c for c in gdelt.columns if c not in ['iso3','year']]}")

    if macro is None:
        print("No macro data — skipping TVTP")
        state_df = baseline_state_df
        kw_final = kw_base
    else:
        gdelt_event_cols = ["protest_count", "conflict_count", "repression_count", "avg_goldstein", "avg_tone"]
        for gc in gdelt_event_cols:
            lag_col = f"{gc}_lag1"
            if gc in macro.columns:
                macro[lag_col] = macro.groupby("iso3")[gc].shift(1)

        all_cov_cols = ["gdp_pc", "gdp_growth", "urbanization", "resource_rents", "trade_openness",
                        "military_spending"] + gdelt_event_cols + [f"{gc}_lag1" for gc in gdelt_event_cols]
        available = [c for c in all_cov_cols if c in macro.columns]
        baseline_state_df_train = baseline_state_df[baseline_state_df["year"] <= MAX_TRAIN_YEAR]
        selected_covs = lasso_select(baseline_state_df_train, macro, available)[:3]

        gdelt_in_top3 = any(gc in selected_covs or f"{gc}_lag1" in selected_covs for gc in gdelt_event_cols)
        if not gdelt_in_top3:
            best_gdelt = None
            for gc in [f"{g}_lag1" for g in gdelt_event_cols] + gdelt_event_cols:
                if gc in available and gc not in selected_covs:
                    best_gdelt = gc
                    break
            if best_gdelt:
                print(f"  GDELT forced inclusion: adding {best_gdelt} as 4th covariate")
                selected_covs = selected_covs[:2] + [best_gdelt]

        if not selected_covs:
            print("No covariates selected — using baseline")
            state_df = baseline_state_df
            kw_final = kw_base
        else:
            print(f"\nBuilding TVTP with top 3 covariates: {selected_covs}")

            macro_sub = macro[["iso3", "year"] + selected_covs].copy()
            for c in selected_covs:
                macro_sub[c] = macro_sub[c].fillna(macro_sub[c].median())
            scaler = StandardScaler()
            macro_sub[selected_covs] = scaler.fit_transform(macro_sub[selected_covs])

            macro_sub = macro_sub.drop_duplicates(subset=["iso3", "year"])
            macro_lookup = macro_sub.set_index(["iso3", "year"])[selected_covs]

            Z_seqs = []
            for i, country in enumerate(country_order):
                cdf = df[df["country_name"] == country].sort_values("year")
                iso3 = cdf["country_text_id"].iloc[0]
                years = cdf["year"].values[-lengths[i]:]
                Z = np.zeros((lengths[i], len(selected_covs)))
                for t, yr in enumerate(years):
                    key = (iso3, yr)
                    if key in macro_lookup.index:
                        Z[t] = macro_lookup.loc[key].values
                Z_seqs.append(Z)

            emit_seqs_train = []
            Z_seqs_train = []
            for i, country in enumerate(country_order):
                cdf = df[df["country_name"] == country].sort_values("year")
                years_full = cdf["year"].values[-lengths[i]:]
                n_pre = int((years_full <= MAX_TRAIN_YEAR).sum())
                if n_pre >= 2:
                    emit_seqs_train.append(emit_seqs[i][:n_pre])
                    Z_seqs_train.append(Z_seqs[i][:n_pre])

            theta = fit_tvtp(emit_seqs_train, Z_seqs_train, baseline, len(selected_covs))

            state_df, tvtp_ll = decode_all(
                emit_seqs, Z_seqs, lengths, country_order, df, baseline, theta,
            )

            print(f"\nTVTP state distribution:")
            for label, count in state_df["state_label"].value_counts().items():
                print(f"  {label}: {count} ({count / len(state_df) * 100:.1f}%)")
            _, kw_final = validate(state_df)

            print(f"\nTVTP adjacent transition effects:")
            for idx_pair, (i, j) in enumerate(ADJ_PAIRS):
                coeffs = ", ".join(f"{selected_covs[c]}={theta[idx_pair, c]:+.3f}" for c in range(len(selected_covs)))
                print(f"  {STATE_LABELS[i][:8]} -> {STATE_LABELS[j][:8]}: {coeffs}")

    output_dir = os.path.dirname(os.path.abspath(__file__))
    state_df.to_csv(os.path.join(output_dir, "country_year_states.csv"), index=False)
    print(f"\nSaved to stage3_msvar/country_year_states.csv")

    return state_df, kw_final


if __name__ == "__main__":
    state_df, kw = run_stage3()

    print("\n=== Sample trajectories ===")
    for country in ["Hungary", "Türkiye", "Poland", "Denmark",
                     "United States of America", "Venezuela", "Tunisia", "Georgia"]:
        sub = state_df[state_df["country_name"] == country].sort_values("year")
        if len(sub) == 0:
            continue
        recent = sub.tail(10)
        traj = ", ".join(f"{int(r['year'])}:{r['state_label'][:4]}" for _, r in recent.iterrows())
        print(f"\n{country}: {traj}")

    print("\n=== Transitions 2020-2025 ===")
    for country in sorted(state_df["country_name"].unique()):
        sub = state_df[(state_df["country_name"] == country) & (state_df["year"] >= 2015)].sort_values("year")
        if len(sub) < 2:
            continue
        states = sub["state"].values
        for t_idx in np.where(np.diff(states) != 0)[0]:
            yr = int(sub["year"].iloc[t_idx + 1])
            if yr >= 2020:
                print(f"  {country}: {STATE_LABELS[states[t_idx]]} -> {STATE_LABELS[states[t_idx + 1]]} ({yr})")
