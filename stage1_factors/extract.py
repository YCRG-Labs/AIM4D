import sys
import os
import re
import numpy as np
import pandas as pd
from scipy import linalg
from scipy.spatial.transform import Rotation
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from data.download_vdem import load_vdem

EXCLUDE_SUFFIXES = (
    "_codehigh", "_codelow", "_sd", "_osp", "_ord",
    "_nr", "_mean", "_3C", "_4C", "_5C",
)
MIN_YEAR = 1970
MAX_MISSING = 0.2
ROW_THRESH = 0.8
K_MAX = 20

FACTOR_LABELS = {
    0: "institutional_entrenchment",
    1: "executive_overreach",
    2: "civil_society_suppression",
    3: "informational_autocratization",
}


def select_indicators(df):
    v2_cols = [c for c in df.columns if c.startswith("v2")]
    filtered = [c for c in v2_cols
                if not any(c.endswith(s) for s in EXCLUDE_SUFFIXES)
                and not re.search(r"_\d+$", c)]

    panel = df[df["year"] >= MIN_YEAR]
    for c in filtered[:]:
        panel[c] = pd.to_numeric(panel[c], errors="coerce")
    all_null = panel[filtered].isnull().all()
    filtered = [c for c in filtered if not all_null[c]]
    missing_pct = panel[filtered].isnull().mean()
    good = missing_pct[missing_pct < MAX_MISSING].index.tolist()

    print(f"Selected {len(good)} indicators (from {len(filtered)} candidates)")
    return good


def build_panel(df, indicators, min_year=MIN_YEAR):
    panel = df[df["year"] >= min_year][
        ["country_name", "country_text_id", "year"] + indicators
    ].copy()

    panel = panel.dropna(subset=indicators, thresh=int(len(indicators) * ROW_THRESH))

    for col in indicators:
        panel[col] = pd.to_numeric(panel[col], errors="coerce")
        panel[col] = panel.groupby("country_name")[col].transform(
            lambda x: x.infer_objects(copy=False).interpolate(limit_direction="both")
        )

    panel = panel.dropna(subset=indicators)

    print(f"Panel: {panel['country_name'].nunique()} countries, "
          f"{panel['year'].min()}-{panel['year'].max()}, "
          f"{len(panel)} country-years, {len(indicators)} indicators")
    return panel


def panel_to_matrix(panel, indicators):
    X_raw = panel[indicators].values
    scaler = StandardScaler()
    X = scaler.fit_transform(X_raw)
    print(f"Matrix: {X.shape[0]} obs x {X.shape[1]} indicators")
    return X, scaler


def bai_ng_ic(X, k_max=K_MAX):
    N, P = X.shape

    cov = X.T @ X / N
    eigenvalues, eigenvectors = linalg.eigh(cov)
    idx = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]

    ic_vals = {1: [], 2: [], 3: []}

    for k in range(1, k_max + 1):
        V_hat = eigenvectors[:, :k]
        X_hat = X @ V_hat @ V_hat.T
        V_k = np.mean((X - X_hat) ** 2)

        CNT = min(N, P)
        ic_vals[1].append(np.log(V_k) + k * ((N + P) / (N * P)) * np.log((N * P) / (N + P)))
        ic_vals[2].append(np.log(V_k) + k * ((N + P) / (N * P)) * np.log(CNT))
        ic_vals[3].append(np.log(V_k) + k * np.log(CNT) / CNT)

    results = {}
    for i in [1, 2, 3]:
        results[i] = np.argmin(ic_vals[i]) + 1

    var_ratios = eigenvalues[:k_max] / eigenvalues[:k_max].sum()
    diffs = np.diff(var_ratios)
    second_diffs = np.diff(diffs)
    elbow_k = np.argmax(second_diffs) + 2 if len(second_diffs) > 0 else 4

    all_maxed = all(results[i] == k_max for i in [1, 2, 3])
    if all_maxed:
        results["elbow"] = max(elbow_k, 4)
        print(f"Bai-Ng: IC1={results[1]}, IC2={results[2]}, IC3={results[3]} (all maxed)")
        print(f"Scree elbow: K={results['elbow']}")
    else:
        results["elbow"] = elbow_k
        print(f"Bai-Ng: IC1={results[1]}, IC2={results[2]}, IC3={results[3]}")

    return results, ic_vals, eigenvalues[:k_max]


def varimax(loadings, max_iter=500, tol=1e-6):
    p, k = loadings.shape
    R = np.eye(k)

    for _ in range(max_iter):
        old_R = R.copy()
        L = loadings @ R
        L2 = L ** 2
        M = loadings.T @ (L2 * L - L * np.mean(L2, axis=0)[None, :])
        U, S, Vt = np.linalg.svd(M)
        R = U @ Vt
        if np.max(np.abs(R - old_R)) < tol:
            break

    return loadings @ R, R


def poet_estimate(X, K):
    N, P = X.shape

    cov = X.T @ X / N
    eigenvalues, eigenvectors = linalg.eigh(cov)
    idx = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]

    raw_loadings = eigenvectors[:, :K] * np.sqrt(eigenvalues[:K])[None, :]
    rotated_loadings, R = varimax(raw_loadings)

    factors = X @ np.linalg.lstsq(rotated_loadings.T @ rotated_loadings,
                                    rotated_loadings.T, rcond=None)[0].T

    low_rank = rotated_loadings @ rotated_loadings.T / P

    residuals = X - factors @ rotated_loadings.T
    R_cov = residuals.T @ residuals / N

    tau = np.zeros((P, P))
    for i in range(P):
        for j in range(i, P):
            t = np.sqrt(np.mean((residuals[:, i] * residuals[:, j] - R_cov[i, j]) ** 2))
            tau[i, j] = t
            tau[j, i] = t

    C = 0.5
    threshold = C * tau * np.sqrt(np.log(P) / N)
    R_thresh = np.sign(R_cov) * np.maximum(np.abs(R_cov) - threshold, 0)
    np.fill_diagonal(R_thresh, np.diag(R_cov))

    Sigma = low_rank + R_thresh

    return {
        "covariance": Sigma,
        "factors": factors,
        "loadings": rotated_loadings,
        "rotation": R,
        "eigenvalues": eigenvalues[:K],
        "K": K,
    }


def label_factors(loading_df, K):
    labels = {}
    for i in range(K):
        col = f"factor_{i+1}"
        top = loading_df[col].abs().nlargest(10).index.tolist()
        label = FACTOR_LABELS.get(i, f"factor_{i+1}")
        labels[col] = {"label": label, "top_indicators": top}
    return labels


def extract_factors(min_year=MIN_YEAR, k_max=K_MAX):
    df = load_vdem()
    indicators = select_indicators(df)
    panel = build_panel(df, indicators, min_year)
    X, scaler = panel_to_matrix(panel, indicators)

    ic_results, ic_vals, top_eigs = bai_ng_ic(X, k_max)
    all_maxed = all(ic_results[i] == k_max for i in [1, 2, 3])
    if all_maxed:
        K = ic_results["elbow"]
        print(f"\nUsing K={K} factors (scree elbow — IC criteria maxed at {k_max})")
    else:
        K = ic_results[2]
        print(f"\nUsing K={K} factors (IC2)")

    result = poet_estimate(X, K)

    var_explained = result["eigenvalues"] / np.trace(X.T @ X / X.shape[0])
    cumulative = np.cumsum(var_explained)
    print(f"Variance explained: {np.round(var_explained * 100, 1)}%")
    print(f"Cumulative: {np.round(cumulative * 100, 1)}%")

    factor_cols = [f"factor_{i+1}" for i in range(K)]
    factor_df = panel[["country_name", "country_text_id", "year"]].copy()
    factor_df[factor_cols] = result["factors"]
    factor_df = factor_df.reset_index(drop=True)

    loading_df = pd.DataFrame(
        result["loadings"],
        index=indicators,
        columns=factor_cols,
    )
    loading_df.index.name = "indicator"

    factor_labels = label_factors(loading_df, K)
    print("\nFactor interpretations:")
    for col, info in factor_labels.items():
        print(f"  {col} -> {info['label']}")
        print(f"    Top: {info['top_indicators'][:5]}")

    output_dir = os.path.dirname(os.path.abspath(__file__))
    factor_df.to_csv(os.path.join(output_dir, "country_year_factors.csv"), index=False)
    loading_df.to_csv(os.path.join(output_dir, "factor_loadings.csv"))
    print(f"\nSaved to stage1_factors/")

    return result, factor_df, loading_df, panel


if __name__ == "__main__":
    result, factor_df, loading_df, panel = extract_factors()

    print("\n=== Factor loadings (top 5 per factor) ===")
    for col in loading_df.columns:
        top = loading_df[col].abs().nlargest(5)
        print(f"\n{col}:")
        for ind, val in top.items():
            print(f"  {ind}: {loading_df.loc[ind, col]:.3f}")

    print("\n=== Sample country trajectories (Factor 1) ===")
    for country in ["Hungary", "Turkey", "Poland", "Denmark", "United States of America"]:
        sub = factor_df[factor_df["country_name"] == country].sort_values("year")
        if len(sub) == 0:
            continue
        recent = sub.tail(5)[["year", "factor_1"]].values
        print(f"\n{country}:")
        for y, v in recent:
            print(f"  {int(y)}: {v:.3f}")
