"""Stage-1 factor robustness beyond the parallel-analysis null in sanity_checks.

  loading_stability : bootstrap percentile CIs on per-factor Tucker congruence
                      between resampled and full-sample loadings. (A bias-corrected
                      and accelerated interval is ill-posed here because the
                      reference statistic is self-congruence, which is identically
                      one and sends the BCa bias term to infinity; the percentile
                      bootstrap is the standard factor-stability interval.)
  split_half        : per-factor congruence between factor models fit on the early
                      and late halves of the panel.
  convergent        : Spearman correlation of each factor score with Polity2, a
                      democracy measure external to V-Dem.
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
from scipy import stats

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from stage1_factors.extract import select_indicators, build_panel, panel_to_matrix
from external_benchmarks import _load_vdem
from sanity_checks import fast_loadings

OUTPUT_DIR = os.path.dirname(os.path.abspath(__file__))
RNG = np.random.default_rng(42)
K = 4
N_BOOT = 200


def per_factor_congruence(ref, other):
    used, out = set(), []
    for a in range(ref.shape[1]):
        best, bj = 0.0, None
        for b in range(other.shape[1]):
            if b in used:
                continue
            den = np.linalg.norm(ref[:, a]) * np.linalg.norm(other[:, b])
            phi = abs(ref[:, a] @ other[:, b]) / den if den > 0 else 0.0
            if phi > best:
                best, bj = phi, b
        if bj is not None:
            used.add(bj)
        out.append(best)
    return np.array(out)


def main():
    print("=" * 70)
    print("STAGE 1 FACTOR ROBUSTNESS")
    print("=" * 70)
    vdem = _load_vdem().copy()
    inds = select_indicators(vdem)
    panel = build_panel(vdem, inds)
    X, _ = panel_to_matrix(panel, inds)
    years = panel["year"].values
    L_full = fast_loadings(X, K)[0]
    out = {}

    print("\n[A] Bootstrap loading stability (per-factor Tucker congruence to full sample)")
    boot = np.array([per_factor_congruence(L_full, fast_loadings(X[RNG.integers(0, len(X), len(X))], K)[0])
                     for _ in range(N_BOOT)])
    for k in range(K):
        lo, md, hi = np.percentile(boot[:, k], [2.5, 50, 97.5])
        print(f"    factor {k+1}: median {md:.3f}, 95% CI [{lo:.3f}, {hi:.3f}]")
        out[f"f{k+1}_boot_med"] = md
        out[f"f{k+1}_boot_lo"] = lo
        out[f"f{k+1}_boot_hi"] = hi

    print("\n[B] Temporal split-half congruence (early vs late panel)")
    mid = np.median(years)
    La = fast_loadings(X[years <= mid], K)[0]
    Lb = fast_loadings(X[years > mid], K)[0]
    split = per_factor_congruence(La, Lb)
    for k in range(K):
        print(f"    factor {k+1}: split-half congruence {split[k]:.3f}")
        out[f"f{k+1}_splithalf"] = split[k]

    print("\n[C] Convergent validity: factor scores vs Polity2 (external to V-Dem)")
    fac = pd.read_csv(os.path.join(REPO, "stage1_factors", "country_year_factors.csv"))
    cow = vdem[["country_text_id", "COWcode"]].drop_duplicates()
    pol = pd.read_csv(os.path.join(REPO, "data", "polity5.csv"))
    pol = pol[(pol["polity2"].notna())][["ccode", "year", "polity2"]].rename(columns={"ccode": "COWcode"})
    m = fac.merge(cow, on="country_text_id", how="left").merge(pol, on=["COWcode", "year"], how="inner")
    for k in range(K):
        col = f"factor_{k+1}"
        if col in m.columns:
            rho, _ = stats.spearmanr(m[col], m["polity2"], nan_policy="omit")
            print(f"    {col} vs Polity2: Spearman rho = {rho:+.3f}  (n={m[col].notna().sum()})")
            out[f"f{k+1}_polity_rho"] = rho

    pd.DataFrame([out]).to_csv(os.path.join(OUTPUT_DIR, "factor_robustness_results.csv"), index=False)
    print(f"\nSaved to robustness/factor_robustness_results.csv")


if __name__ == "__main__":
    main()
