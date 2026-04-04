import sys
import os
import numpy as np
import pandas as pd
from scipy import linalg
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from data.download_vdem import load_vdem

CORE_INDICES = [
    "v2x_polyarchy", "v2x_libdem", "v2x_partipdem", "v2x_delibdem", "v2x_egaldem",
    "v2x_liberal", "v2x_partip", "v2x_egal",
    "v2x_accountability", "v2x_rule", "v2x_corr", "v2x_pubcorr", "v2x_execorr",
    "v2x_neopat", "v2x_civlib", "v2x_clphy", "v2x_clpol", "v2x_clpriv",
    "v2x_freexp_altinf", "v2x_frassoc_thick", "v2x_suffr",
    "v2x_jucon", "v2x_cspart", "v2x_mpi",
    "v2x_gender", "v2x_gencl", "v2x_gencs", "v2x_genpp",
    "v2x_elecoff", "v2x_EDcomp_thick",
    "v2x_api", "v2x_diagacc", "v2x_horacc", "v2x_veracc",
]

MIN_YEAR = 1970
K_MAX = 15


def build_panel(df, min_year=MIN_YEAR):
    available = [c for c in CORE_INDICES if c in df.columns]
    panel = df[df["year"] >= min_year][["country_name", "country_text_id", "year"] + available].copy()
    panel = panel.dropna(subset=available, thresh=len(available) * 0.8)

    for col in available:
        panel[col] = panel.groupby("country_name")[col].transform(
            lambda x: x.interpolate(limit_direction="both")
        )
    panel = panel.dropna(subset=available)

    print(f"Panel: {panel['country_name'].nunique()} countries, "
          f"{panel['year'].min()}-{panel['year'].max()}, "
          f"{len(available)} indices")
    return panel, available


def panel_to_matrix(panel, indicators):
    grouped = panel.groupby("country_name")[indicators].mean()
    grouped = grouped.dropna()

    scaler = StandardScaler()
    X = scaler.fit_transform(grouped)

    print(f"Matrix: {X.shape[0]} countries x {X.shape[1]} indicators")
    return X, grouped.index.tolist(), scaler


def bai_ng_ic(X, k_max=K_MAX):
    N, P = X.shape

    cov = X.T @ X / N
    eigenvalues, eigenvectors = linalg.eigh(cov)
    idx = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[idx]

    ic_vals = {1: [], 2: [], 3: []}

    for k in range(1, k_max + 1):
        V_hat = eigenvectors[:, idx[:k]]
        X_hat = X @ V_hat @ V_hat.T
        V_k = np.mean((X - X_hat) ** 2)

        ic_vals[1].append(np.log(V_k) + k * ((N + P) / (N * P)) * np.log((N * P) / (N + P)))
        ic_vals[2].append(np.log(V_k) + k * ((N + P) / (N * P)) * np.log(min(N, P)))
        ic_vals[3].append(np.log(V_k) + k * np.log(min(N, P)) / min(N, P))

    results = {}
    for i in [1, 2, 3]:
        results[i] = np.argmin(ic_vals[i]) + 1

    print(f"Bai-Ng: IC1={results[1]}, IC2={results[2]}, IC3={results[3]}")
    return results, ic_vals


def poet_estimate(X, K):
    N, P = X.shape

    cov = X.T @ X / N
    eigenvalues, eigenvectors = linalg.eigh(cov)
    idx = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]

    loadings = eigenvectors[:, :K] * np.sqrt(P)
    factors = X @ loadings / P

    low_rank = loadings @ loadings.T / P

    residuals = X - factors @ loadings.T
    R = residuals.T @ residuals / N

    tau = np.zeros((P, P))
    for i in range(P):
        for j in range(i, P):
            t = np.sqrt(np.mean((residuals[:, i] * residuals[:, j] - R[i, j]) ** 2))
            tau[i, j] = t
            tau[j, i] = t

    C = 0.5
    threshold = C * tau * np.sqrt(np.log(P) / N)
    R_thresh = np.sign(R) * np.maximum(np.abs(R) - threshold, 0)
    np.fill_diagonal(R_thresh, np.diag(R))

    Sigma = low_rank + R_thresh

    return {
        "covariance": Sigma,
        "factors": factors,
        "loadings": loadings,
        "eigenvalues": eigenvalues[:K],
        "K": K,
    }


def extract_factors(min_year=MIN_YEAR, k_max=K_MAX):
    df = load_vdem()
    panel, indicators = build_panel(df, min_year)
    X, countries, scaler = panel_to_matrix(panel, indicators)

    ic_results, ic_vals = bai_ng_ic(X, k_max)
    K = ic_results[1]
    print(f"\nUsing K={K} factors")

    result = poet_estimate(X, K)

    var_explained = result["eigenvalues"] / np.trace(X.T @ X / X.shape[0])
    cumulative = np.cumsum(var_explained)
    print(f"Variance explained: {np.round(var_explained * 100, 1)}%")
    print(f"Cumulative: {np.round(cumulative * 100, 1)}%")

    factor_df = pd.DataFrame(
        result["factors"],
        index=countries,
        columns=[f"factor_{i+1}" for i in range(K)]
    )
    factor_df.index.name = "country_name"

    loading_df = pd.DataFrame(
        result["loadings"],
        index=indicators,
        columns=[f"factor_{i+1}" for i in range(K)]
    )
    loading_df.index.name = "indicator"

    output_dir = os.path.dirname(os.path.abspath(__file__))
    factor_df.to_csv(os.path.join(output_dir, "country_factors.csv"))
    loading_df.to_csv(os.path.join(output_dir, "factor_loadings.csv"))
    print(f"\nSaved to stage1_factors/")

    return result, factor_df, loading_df, panel


if __name__ == "__main__":
    result, factor_df, loading_df, panel = extract_factors()

    print("\n=== Factor 1: Top loadings ===")
    f1 = loading_df["factor_1"].abs().sort_values(ascending=False)
    print(f1.head(10))

    print("\n=== Countries by Factor 1 ===")
    f1_countries = factor_df["factor_1"].sort_values()
    print("Bottom 5:", list(f1_countries.head().index))
    print("Top 5:", list(f1_countries.tail().index))
