import sys
import os
import warnings
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, cohen_kappa_score
from sklearn.linear_model import LogisticRegressionCV
from scipy.optimize import minimize
from scipy.special import softmax, logsumexp
from hmmlearn import hmm

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

FACTOR_COLS = ["factor_1", "factor_2", "factor_3", "factor_4"]
N_STATES = 5
N_RESTARTS = 50
DIRICHLET_DIAG = 50
DIRICHLET_OFF = 2
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
    return factors


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


def kmeans_init(X_all, n_states=N_STATES):
    km = KMeans(n_clusters=n_states, n_init=30, random_state=42)
    km.fit(X_all)

    order = np.argsort(-km.cluster_centers_[:, 0])
    means = km.cluster_centers_[order]
    covars = []
    for k in range(n_states):
        mask = km.labels_ == order[k]
        if mask.sum() > X_all.shape[1] + 1:
            cov = np.cov(X_all[mask].T) + 1e-4 * np.eye(X_all.shape[1])
        else:
            cov = np.eye(X_all.shape[1])
        covars.append(cov)

    print(f"K-means init (no V-Dem supervision):")
    for s in range(n_states):
        n = (km.labels_ == order[s]).sum()
        print(f"  Cluster {s} ({STATE_LABELS[s]}): {n} obs, F1 mean={means[s, 0]:.3f}")

    return means, np.array(covars)


def regularize_transmat(P, n_states=N_STATES):
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


def fit_baseline_hmm(X_all, lengths, init_means, init_covars):
    init_transmat = np.full((N_STATES, N_STATES), 0.005)
    for i in range(N_STATES):
        init_transmat[i, i] = 0.95
        if i > 0:
            init_transmat[i, i - 1] = 0.02
        if i < N_STATES - 1:
            init_transmat[i, i + 1] = 0.02
    init_transmat /= init_transmat.sum(axis=1, keepdims=True)

    best_model = None
    best_score = -np.inf

    for restart in range(N_RESTARTS):
        model = hmm.GaussianHMM(
            n_components=N_STATES,
            covariance_type="full",
            n_iter=500,
            tol=1e-5,
            random_state=restart,
            init_params="",
        )

        rng = np.random.RandomState(restart)
        scale = 0.1 if restart < N_RESTARTS // 2 else 0.3
        model.means_ = init_means + rng.randn(*init_means.shape) * scale if restart > 0 else init_means.copy()
        model.covars_ = init_covars.copy()
        perturbed = init_transmat + rng.dirichlet(np.ones(N_STATES) * 10, size=N_STATES) * 0.05 if restart > 0 else init_transmat.copy()
        perturbed /= perturbed.sum(axis=1, keepdims=True)
        model.transmat_ = perturbed
        model.startprob_ = np.ones(N_STATES) / N_STATES

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            try:
                model.fit(X_all, lengths)
                score = model.score(X_all, lengths)
                if np.all(np.diff(model.means_[:, 0]) <= 0) and score > best_score:
                    best_score = score
                    best_model = model
            except Exception:
                continue

        if (restart + 1) % 20 == 0:
            status = f"score={best_score:.1f}" if best_model else "no valid model"
            print(f"  Restart {restart+1}/{N_RESTARTS}: {status}")

    if best_model is None:
        print("  Fallback: unconstrained + reorder")
        for restart in range(N_RESTARTS):
            model = hmm.GaussianHMM(
                n_components=N_STATES, covariance_type="full",
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

    for k in range(K):
        diff = X_all - means[k]
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

    df = load_inputs()
    obs_cols = FACTOR_COLS
    print(f"Features: {obs_cols}")

    X_all, lengths, country_order = prepare_sequences(df, obs_cols)
    print(f"Panel: {len(country_order)} countries, {sum(lengths)} obs\n")

    init_means, init_covars = kmeans_init(X_all)

    print(f"\nPhase 1: Baseline HMM (K-means init, no supervision)...")
    baseline, base_score = fit_baseline_hmm(X_all, lengths, init_means, init_covars)

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
    if macro is None:
        print("No macro data — skipping TVTP")
        state_df = baseline_state_df
        kw_final = kw_base
    else:
        all_cov_cols = ["gdp_pc", "gdp_growth", "urbanization", "resource_rents", "trade_openness", "military_spending"]
        available = [c for c in all_cov_cols if c in macro.columns]
        selected_covs = lasso_select(baseline_state_df, macro, available)[:3]

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

            theta = fit_tvtp(emit_seqs, Z_seqs, baseline, len(selected_covs))

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
