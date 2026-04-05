import sys
import os
import warnings
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegressionCV
from sklearn.metrics import confusion_matrix, cohen_kappa_score
from hmmlearn import hmm

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

FACTOR_COLS = ["factor_1", "factor_2", "factor_3", "factor_4"]
N_STATES = 5
N_RESTARTS = 50
DIRICHLET_PERSISTENCE = 50
DIRICHLET_OFF_DIAG = 2
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
    return merged, FACTOR_COLS, beta_cols


def add_momentum_features(df, factor_cols):
    momentum_cols = []
    for col in factor_cols:
        dcol = f"d_{col}"
        df[dcol] = df.groupby("country_name")[col].diff()
        momentum_cols.append(dcol)
    return momentum_cols


def load_vdem_regime():
    vdem = pd.read_csv(
        os.path.join(os.path.dirname(__file__), "..", "data", "vdem_v16.csv"),
        low_memory=False,
        usecols=["country_name", "year", "v2x_regime", "v2x_polyarchy"],
    )
    vdem = vdem.dropna(subset=["v2x_regime"])
    vdem["v2x_regime"] = vdem["v2x_regime"].astype(int)
    return vdem


def map_to_5_states(vdem_regime, polyarchy):
    if vdem_regime == 3:
        return 0
    elif vdem_regime == 2:
        return 1
    elif vdem_regime == 1:
        if polyarchy is not None and polyarchy > 0.35:
            return 2
        else:
            return 3
    else:
        return 4


def semi_supervised_init(df, obs_cols, vdem):
    merged = df.merge(vdem[["country_name", "year", "v2x_regime", "v2x_polyarchy"]],
                      on=["country_name", "year"], how="left")
    merged["init_state"] = merged.apply(
        lambda r: map_to_5_states(r["v2x_regime"], r["v2x_polyarchy"])
        if pd.notna(r["v2x_regime"]) else -1, axis=1
    )

    X = df[obs_cols].values
    init_means = np.zeros((N_STATES, len(obs_cols)))
    init_covars = np.zeros((N_STATES, len(obs_cols), len(obs_cols)))

    for s in range(N_STATES):
        mask = merged["init_state"].values == s
        if mask.sum() > len(obs_cols) + 1:
            init_means[s] = X[mask].mean(axis=0)
            init_covars[s] = np.cov(X[mask].T) + 1e-4 * np.eye(len(obs_cols))
        else:
            init_means[s] = X.mean(axis=0) + np.random.randn(len(obs_cols)) * 0.1
            init_covars[s] = np.eye(len(obs_cols))

    for s in range(N_STATES):
        n = (merged["init_state"].values == s).sum()
        print(f"  Init state {s} ({STATE_LABELS[s]}): {n} obs, F1 mean={init_means[s, 0]:.3f}")

    return init_means, init_covars


def prepare_sequences(df, obs_cols):
    countries = df["country_name"].unique()
    sequences = []
    lengths = []
    country_order = []

    for country in sorted(countries):
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


def regularize_transmat(P, n_states, alpha_diag=DIRICHLET_PERSISTENCE, alpha_off=DIRICHLET_OFF_DIAG):
    alpha = np.full((n_states, n_states), alpha_off)
    np.fill_diagonal(alpha, alpha_diag)

    for i in range(n_states):
        for j in range(n_states):
            if abs(i - j) > 2:
                alpha[i, j] = 0.1

    counts = P * 100
    smoothed = counts + alpha
    smoothed /= smoothed.sum(axis=1, keepdims=True)
    return smoothed


def fit_hmm(X_all, lengths, init_means, init_covars, n_states=N_STATES, n_restarts=N_RESTARTS):
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

    for restart in range(n_restarts):
        model = hmm.GaussianHMM(
            n_components=n_states,
            covariance_type="full",
            n_iter=500,
            tol=1e-5,
            random_state=restart,
            init_params="",
        )

        noise = np.random.RandomState(restart)
        if restart == 0:
            model.means_ = init_means.copy()
            model.covars_ = init_covars.copy()
        else:
            scale = 0.1 if restart < n_restarts // 2 else 0.3
            model.means_ = init_means + noise.randn(*init_means.shape) * scale
            model.covars_ = init_covars.copy()

        perturbed = init_transmat.copy()
        if restart > 0:
            perturbed += noise.dirichlet(np.ones(n_states) * 10, size=n_states) * 0.05
            perturbed /= perturbed.sum(axis=1, keepdims=True)
        model.transmat_ = perturbed
        model.startprob_ = np.ones(n_states) / n_states

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            try:
                model.fit(X_all, lengths)
                score = model.score(X_all, lengths)

                mean_f1 = model.means_[:, 0]
                if np.all(np.diff(mean_f1) <= 0) and score > best_score:
                    best_score = score
                    best_model = model
            except Exception:
                continue

        if (restart + 1) % 20 == 0:
            status = f"score={best_score:.1f}" if best_model else "no valid model"
            print(f"  Restart {restart+1}/{n_restarts}: {status}")

    if best_model is None:
        print("  Falling back to unconstrained best model...")
        for restart in range(n_restarts):
            model = hmm.GaussianHMM(
                n_components=n_states, covariance_type="full",
                n_iter=500, tol=1e-5, random_state=restart + 5000,
            )
            model.means_ = init_means + np.random.RandomState(restart + 5000).randn(*init_means.shape) * 0.2
            model.covars_ = init_covars.copy()
            model.transmat_ = init_transmat.copy()
            model.startprob_ = np.ones(n_states) / n_states
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

    print(f"\nBest log-likelihood: {best_score:.1f}")

    raw_transmat = best_model.transmat_.copy()
    best_model.transmat_ = regularize_transmat(raw_transmat, n_states)
    print(f"\nTransition matrix regularized (Dirichlet α_diag={DIRICHLET_PERSISTENCE}, α_off={DIRICHLET_OFF_DIAG})")
    print(f"  Min diagonal persistence: {np.min(np.diag(best_model.transmat_)):.3f}")

    return best_model, best_score


def select_k_by_bic(X_all, lengths, init_means_full, init_covars_full):
    results = {}
    for k in [3, 4, 5, 6]:
        model = hmm.GaussianHMM(
            n_components=k, covariance_type="diag",
            n_iter=200, tol=1e-3, random_state=42,
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            try:
                model.fit(X_all, lengths)
                score = model.score(X_all, lengths)
                d = X_all.shape[1]
                n_params = k * d + k * d * (d + 1) / 2 + k * (k - 1)
                bic = -2 * score + n_params * np.log(len(X_all))
                results[k] = {"score": score, "bic": bic, "n_params": int(n_params)}
            except Exception:
                pass

    print("\nModel selection (BIC):")
    for k, r in sorted(results.items()):
        marker = " <--" if k == min(results, key=lambda x: results[x]["bic"]) else ""
        print(f"  K={k}: BIC={r['bic']:.1f}, LL={r['score']:.1f}, params={r['n_params']}{marker}")

    return results


def decode_states(model, X_all, lengths, country_order, df, obs_cols):
    posteriors = model.predict_proba(X_all, lengths)
    states = model.predict(X_all, lengths)

    rows = []
    idx = 0
    for i, country in enumerate(country_order):
        T = lengths[i]
        cdf = df[df["country_name"] == country].sort_values("year")
        years = cdf["year"].values

        for t in range(T):
            row = {
                "country_name": country,
                "country_text_id": cdf["country_text_id"].iloc[0],
                "year": int(years[t]),
                "state": int(states[idx]),
                "state_label": STATE_LABELS.get(int(states[idx]), f"state_{states[idx]}"),
            }
            for s in range(model.n_components):
                row[f"prob_state_{s}"] = posteriors[idx, s]
            rows.append(row)
            idx += 1

    return pd.DataFrame(rows)


def validate_against_vdem(state_df, vdem):
    merged = state_df.merge(vdem[["country_name", "year", "v2x_regime", "v2x_polyarchy"]],
                            on=["country_name", "year"], how="inner")

    merged["vdem_5state"] = merged.apply(
        lambda r: map_to_5_states(r["v2x_regime"], r["v2x_polyarchy"]), axis=1
    )

    our = merged["state"].values
    ref = merged["vdem_5state"].values

    kappa = cohen_kappa_score(ref, our)
    accuracy = np.mean(ref == our)

    kappa_w = cohen_kappa_score(ref, our, weights="linear")

    print(f"\n=== Validation vs V-Dem (5-state) ===")
    print(f"Cohen's kappa: {kappa:.3f}")
    print(f"Weighted kappa (linear): {kappa_w:.3f}")
    print(f"Accuracy: {accuracy:.3f}")

    print(f"\nConfusion matrix (rows=V-Dem 5-state, cols=ours):")
    labels = list(range(N_STATES))
    cm = confusion_matrix(ref, our, labels=labels)
    names_short = ["lib_dem", "elec_dem", "hybrid", "comp_auth", "clos_auth"]
    print(pd.DataFrame(cm, index=names_short, columns=names_short))

    print(f"\nPer-state accuracy:")
    for s in range(N_STATES):
        mask = ref == s
        if mask.sum() > 0:
            acc = np.mean(our[mask] == s)
            print(f"  {STATE_LABELS[s]}: {acc:.3f} ({mask.sum()} obs)")

    polyarchy_by_state = merged.groupby("state")["v2x_polyarchy"].agg(["mean", "std", "count"])
    print(f"\nPolyarchy by our state:")
    for s in range(N_STATES):
        if s in polyarchy_by_state.index:
            r = polyarchy_by_state.loc[s]
            print(f"  State {s} ({STATE_LABELS[s]}): {r['mean']:.3f} ± {r['std']:.3f} (n={int(r['count'])})")

    return kappa, kappa_w


def load_macro_covariates():
    path = os.path.join(os.path.dirname(__file__), "..", "data", "macro_covariates.csv")
    if os.path.exists(path):
        return pd.read_csv(path)

    try:
        import wbgapi as wb
        indicators = {
            "NY.GDP.PCAP.KD": "gdp_pc",
            "NY.GDP.MKTP.KD.ZG": "gdp_growth",
            "NE.TRD.GNFS.ZS": "trade_openness",
            "NY.GDP.TOTL.RT.ZS": "resource_rents",
            "SP.URB.TOTL.IN.ZS": "urbanization",
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
            except Exception as e:
                print(f"  Warning: {name}: {e}")
        if not frames:
            return None
        macro = frames[0]
        for f in frames[1:]:
            macro = macro.merge(f, on=["iso3", "year"], how="outer")
        macro.to_csv(path, index=False)
        return macro
    except ImportError:
        return None


def lasso_covariate_selection(state_df, macro_df):
    if macro_df is None:
        return None

    merged = state_df.merge(
        macro_df, left_on=["country_text_id", "year"],
        right_on=["iso3", "year"], how="inner",
    )
    covariate_cols = [c for c in macro_df.columns if c not in ["iso3", "year"]]
    merged = merged.dropna(subset=covariate_cols + ["state"])

    if len(merged) < 200:
        print(f"Only {len(merged)} obs after merge — skipping LASSO")
        return None

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

    print(f"\nLASSO selected {len(selected)}/{len(covariate_cols)} covariates:")
    for name, importance in selected:
        print(f"  {name}: {importance:.4f}")

    return selected


def run_stage3():
    print("=== Stage 3: Markov-Switching Regime Classification ===\n")

    df, factor_cols, beta_cols = load_inputs()

    obs_cols = factor_cols
    print(f"Observation vector: {len(obs_cols)} features: {obs_cols}")

    vdem = load_vdem_regime()

    X_all, lengths, country_order = prepare_sequences(df, obs_cols)
    print(f"\nPanel: {len(country_order)} countries, {sum(lengths)} obs, {X_all.shape[1]} features")

    print(f"\nSemi-supervised initialization from V-Dem RoW:")
    init_means, init_covars = semi_supervised_init(df, obs_cols, vdem)

    print(f"\nFitting {N_STATES}-state HMM ({N_RESTARTS} restarts, semi-supervised init)...")
    model, score = fit_hmm(X_all, lengths, init_means, init_covars)

    print(f"\nState means:")
    for s in range(N_STATES):
        f_means = ", ".join(f"{model.means_[s, i]:.3f}" for i in range(len(factor_cols)))
        print(f"  State {s} ({STATE_LABELS[s]}): factors=[{f_means}]")

    print(f"\nTransition matrix:")
    P = model.transmat_
    for i in range(N_STATES):
        row = " ".join(f"{P[i,j]:.4f}" for j in range(N_STATES))
        print(f"  {STATE_LABELS[i]:30s} [{row}]")

    state_df = decode_states(model, X_all, lengths, country_order, df, obs_cols)

    print(f"\nState distribution:")
    dist = state_df["state_label"].value_counts()
    for label, count in dist.items():
        print(f"  {label}: {count} ({count/len(state_df)*100:.1f}%)")

    kappa, kappa_w = validate_against_vdem(state_df, vdem)

    output_dir = os.path.dirname(os.path.abspath(__file__))
    state_df.to_csv(os.path.join(output_dir, "country_year_states.csv"), index=False)

    print(f"\n=== Macro covariates & LASSO ===")
    macro = load_macro_covariates()
    if macro is not None:
        lasso_covariate_selection(state_df, macro)

    return model, state_df, kappa_w


if __name__ == "__main__":
    model, state_df, kappa_w = run_stage3()

    print("\n=== Sample trajectories ===")
    for country in ["Hungary", "Türkiye", "Poland", "Denmark",
                     "United States of America", "Venezuela", "Tunisia", "Georgia"]:
        sub = state_df[state_df["country_name"] == country].sort_values("year")
        if len(sub) == 0:
            continue
        recent = sub.tail(10)
        states_str = ", ".join(f"{int(r['year'])}:{r['state_label'][:4]}" for _, r in recent.iterrows())
        print(f"\n{country}: {states_str}")

    print("\n=== Recent transitions (2020-2025) ===")
    for country in sorted(state_df["country_name"].unique()):
        sub = state_df[(state_df["country_name"] == country) & (state_df["year"] >= 2015)].sort_values("year")
        if len(sub) < 2:
            continue
        states = sub["state"].values
        transitions = np.where(np.diff(states) != 0)[0]
        for t_idx in transitions:
            yr = int(sub["year"].iloc[t_idx + 1])
            if yr >= 2020:
                from_s = STATE_LABELS[states[t_idx]]
                to_s = STATE_LABELS[states[t_idx + 1]]
                print(f"  {country}: {from_s} -> {to_s} ({yr})")
