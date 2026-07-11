"""Adebayo-style sanity checks (Adebayo et al. 2018) for the Stage-1 factor and
Stage-3 regime decompositions: confirm they are genuinely data-driven, not
artifacts that survive randomization.

  Factors:  (a) input-permutation -- permute each indicator independently, refit
                POET, show factor loadings collapse (Tucker congruence -> ~chance);
            (b) parallel analysis -- real top-K eigenvalues vs the 95th-percentile
                null from column-permuted data (Horn 1965).
  Regimes:  refit a 5-state Gaussian HMM on the real factor sequences vs on
            factor-scrambled inputs vs random-state assignment, and compare
            agreement with V-Dem Regimes-of-the-World (weighted kappa). Real should
            far exceed both randomized controls.
"""

import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "BLIS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, "4")

import sys
import warnings

warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
from scipy import linalg
from sklearn.metrics import cohen_kappa_score
from hmmlearn import hmm

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from stage1_factors.extract import select_indicators, build_panel, panel_to_matrix, varimax
from external_benchmarks import _load_vdem

OUTPUT_DIR = os.path.dirname(os.path.abspath(__file__))
RNG = np.random.default_rng(42)
K = 4
FACTOR_COLS = ["factor_1", "factor_2", "factor_3", "factor_4"]
N_STATES = 5


def fast_loadings(X, k=K):
    """POET loadings (eigvec * sqrt(eigval), varimax-rotated) without the residual
    covariance thresholding, which does not affect loadings."""
    cov = X.T @ X / X.shape[0]
    evals, evecs = linalg.eigh(cov)
    idx = np.argsort(evals)[::-1]
    raw = evecs[:, idx[:k]] * np.sqrt(np.maximum(evals[idx[:k]], 0))[None, :]
    rotated, _ = varimax(raw)
    return rotated, evals[idx]


def tucker_congruence(A, B):
    """Mean matched Tucker phi between two loading matrices (greedy, sign-invariant)."""
    used = set()
    phis = []
    for a in range(A.shape[1]):
        best, best_j = 0.0, None
        for b in range(B.shape[1]):
            if b in used:
                continue
            num = abs(A[:, a] @ B[:, b])
            den = np.linalg.norm(A[:, a]) * np.linalg.norm(B[:, b])
            phi = num / den if den > 0 else 0.0
            if phi > best:
                best, best_j = phi, b
        if best_j is not None:
            used.add(best_j)
            phis.append(best)
    return float(np.mean(phis))


def factor_sanity(n_perm=20):
    print("\n### Stage 1 factor sanity checks")
    vdem = _load_vdem().copy()
    indicators = select_indicators(vdem)
    panel = build_panel(vdem, indicators)
    X, _ = panel_to_matrix(panel, indicators)
    real_load, real_evals = fast_loadings(X, K)

    perm_congr, perm_top_evals = [], []
    for _ in range(n_perm):
        Xp = np.column_stack([RNG.permutation(X[:, j]) for j in range(X.shape[1])])
        load_p, evals_p = fast_loadings(Xp, K)
        perm_congr.append(tucker_congruence(real_load, load_p))
        perm_top_evals.append(evals_p[:K])
    perm_congr = np.array(perm_congr)
    null_95 = np.percentile(np.array(perm_top_evals), 95, axis=0)

    print(f"  input-permutation Tucker congruence vs real: "
          f"{perm_congr.mean():.3f} +/- {perm_congr.std():.3f}  (real self = 1.000)")
    print("  parallel analysis (Horn): real top-K eigenvalues vs permuted 95th pctile")
    above = 0
    for k in range(K):
        flag = real_evals[k] > null_95[k]
        above += flag
        print(f"    factor {k+1}: real={real_evals[k]:.2f}  null95={null_95[k]:.2f}  "
              f"{'ABOVE (retained)' if flag else 'below'}")
    out = {"factor_perm_congruence_mean": perm_congr.mean(),
           "factor_perm_congruence_sd": perm_congr.std(),
           "factors_above_null": int(above)}
    for k in range(K):
        out[f"eigenvalue_f{k+1}"] = float(real_evals[k])
        out[f"null95_f{k+1}"] = float(null_95[k])
    return out


def build_sequences():
    fac = pd.read_csv(os.path.join(REPO, "stage1_factors", "country_year_factors.csv"))
    fac = fac.sort_values(["country_name", "year"])
    lag_cols = []
    for fc in FACTOR_COLS:
        lc = f"lag_{fc}"
        fac[lc] = fac.groupby("country_name")[fc].shift(1)
        lag_cols.append(lc)
    obs_cols = FACTOR_COLS + lag_cols
    fac = fac.dropna(subset=obs_cols)
    seqs, lengths, order, year_lists = [], [], [], []
    for country, cdf in fac.groupby("country_name"):
        cdf = cdf.sort_values("year")
        if len(cdf) < 10:
            continue
        seqs.append(cdf[obs_cols].values)
        lengths.append(len(cdf))
        order.append(country)
        year_lists.append(cdf["year"].values)
    return np.concatenate(seqs), lengths, order, year_lists


def fit_and_decode(X_all, lengths, order, year_lists, n_restarts=8):
    best, best_ll = None, -np.inf
    for r in range(n_restarts):
        m = hmm.GaussianHMM(n_components=N_STATES, covariance_type="diag",
                            min_covar=0.05, n_iter=200, tol=1e-4, random_state=r)
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                m.fit(X_all, lengths)
                ll = m.score(X_all, lengths)
            if ll > best_ll:
                best_ll, best = ll, m
        except Exception:
            continue
    if best is None:
        return None
    reorder = np.argsort(-best.means_[:, 0])
    remap = {old: new for new, old in enumerate(reorder)}
    rows, idx = [], 0
    for c, L, yrs in zip(order, lengths, year_lists):
        seq = X_all[idx:idx + L]
        states = best.predict(seq)
        for t in range(L):
            rows.append({"country_name": c, "year": int(yrs[t]), "state": remap[states[t]]})
        idx += L
    return pd.DataFrame(rows)


def kappa_vs_row(state_df):
    vdem = pd.read_csv(os.path.join(REPO, "data", "vdem_v16.csv"), low_memory=False,
                       usecols=["country_name", "year", "v2x_regime", "v2x_polyarchy"])
    vdem = vdem.dropna(subset=["v2x_regime"])
    vdem["v2x_regime"] = vdem["v2x_regime"].astype(int)

    def to5(reg, poly):
        if reg == 3: return 0
        if reg == 2: return 1
        if reg == 1: return 2 if (poly is not None and poly > 0.35) else 3
        return 4

    m = state_df.merge(vdem, on=["country_name", "year"], how="inner")
    m["ref"] = [to5(r, p) for r, p in zip(m["v2x_regime"], m["v2x_polyarchy"])]
    return cohen_kappa_score(m["ref"], m["state"], weights="linear")


def regime_sanity():
    print("\n### Stage 3 regime sanity checks (weighted kappa vs V-Dem RoW)")
    X_all, lengths, order, year_lists = build_sequences()

    real = fit_and_decode(X_all, lengths, order, year_lists)
    k_real = kappa_vs_row(real)

    perm = RNG.permutation(X_all.shape[0])
    shuf = fit_and_decode(X_all[perm], lengths, order, year_lists)
    k_shuf = kappa_vs_row(shuf)

    rand_df = real.copy()
    rand_df["state"] = RNG.integers(0, N_STATES, size=len(rand_df))
    k_rand = kappa_vs_row(rand_df)

    print(f"  real factor sequences   : kappa_w = {k_real:.3f}")
    print(f"  scrambled-factor inputs : kappa_w = {k_shuf:.3f}  (data randomization)")
    print(f"  random-state assignment : kappa_w = {k_rand:.3f}  (chance floor)")
    return {"hmm_kappa_real": k_real, "hmm_kappa_scrambled": k_shuf, "hmm_kappa_random": k_rand}


def main():
    print("=" * 70)
    print("SANITY CHECKS: factor + regime decomposition (Adebayo et al. 2018)")
    print("=" * 70)
    r1 = factor_sanity()
    r2 = regime_sanity()
    pd.DataFrame([{**r1, **r2}]).to_csv(os.path.join(OUTPUT_DIR, "sanity_checks_results.csv"), index=False)
    print(f"\nSaved to robustness/sanity_checks_results.csv")


if __name__ == "__main__":
    main()
