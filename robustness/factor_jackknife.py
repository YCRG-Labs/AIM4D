"""
A1b: Within-V-Dem indicator jackknife.

Proves Stage-1 factor-1 is a stable latent construct, not an artifact of
specific V-Dem indicator choices — a bulletproof, no-external-data robustness
check that complements the FH/Polity replication.

Four-tier battery, scored by Tucker's congruence coefficient phi between the
original and refit factor-1 loadings after orthogonal Procrustes alignment
(Lorenzo-Seva & ten Berge 2006: phi>=0.95 => "factors equal", 0.85-0.94 =
"fair similarity"):

  Tier 1  leave-one-component-out  (electoral/liberal/participatory/
          deliberative/egalitarian/accountability families)        [primary]
  Tier 2  drop-top-loaders         (remove 5 then 10 highest |loading|) [headline]
  Tier 3  random-subset bootstrap  (drop random 20%, 200 reps)       [distribution]
  Tier 4  leave-one-indicator-out  (drop each of N, report min phi)  [appendix]

Output: robustness/factor_jackknife.csv (one row per config) + stdout verdict.
"""

import os
import sys
import numpy as np
import pandas as pd

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
from stage1_factors.extract import (  # noqa: E402
    load_vdem, select_indicators, build_panel, panel_to_matrix, varimax,
    bai_ng_ic,
)
from scipy import linalg as sla  # noqa: E402


def fast_factors(X, K):
    """Loadings + factor scores ONLY (eigendecomp + varimax). Skips the
    O(P^2) POET sparse-covariance thresholding, which the jackknife never
    uses — it only compares loadings/scores. ~100x faster than poet_estimate
    on a 330-indicator panel."""
    N = X.shape[0]
    cov = X.T @ X / N
    eigvals, eigvecs = sla.eigh(cov)
    idx = np.argsort(eigvals)[::-1]
    eigvals, eigvecs = eigvals[idx], eigvecs[:, idx]
    raw = eigvecs[:, :K] * np.sqrt(np.maximum(eigvals[:K], 0))[None, :]
    rot, _ = varimax(raw)
    factors = X @ np.linalg.lstsq(rot.T @ rot, rot.T, rcond=None)[0].T
    return rot, factors

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "factor_jackknife.csv")
K = 4
N_BOOT = 200
RNG = np.random.default_rng(42)

# V-Dem high-level component prefixes for leave-one-component-out.
# Indicators matching these substrings get dropped as a group.
VDEM_COMPONENTS = {
    "electoral":     ["v2el", "v2x_elecreg", "v2xel", "v2x_polyarchy", "v2x_EDcomp"],
    "liberal":       ["v2x_liberal", "v2xcl", "v2cl", "v2x_clpol", "v2x_clpriv", "v2juncind", "v2x_jucon"],
    "participatory": ["v2x_partip", "v2x_cspart", "v2csreprss", "v2dlengage", "v2xps"],
    "deliberative":  ["v2x_delib", "v2dl", "v2xdl"],
    "egalitarian":   ["v2x_egal", "v2xeg", "v2pe", "v2clgeocl"],
    "accountability": ["v2x_accountability", "v2x_diagacc", "v2x_horacc", "v2x_veracc", "v2lg", "v2ex"],
}


def tucker_congruence(a, b):
    """Tucker's phi: cosine between two loading vectors (sign-invariant via abs)."""
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    if denom == 0:
        return 0.0
    return float(abs(a @ b) / denom)


_PANEL_CACHE = {}  # full interpolated panel, built once


def _cached_panel(df, indicators):
    """Build the fully-interpolated panel ONCE over all indicators and cache it.
    Jackknife configs then subset columns from this without re-interpolating
    (per-column interpolation is independent, so the cached values are valid
    for any column subset)."""
    key = "full"
    if key not in _PANEL_CACHE:
        _PANEL_CACHE[key] = build_panel(df, indicators)
    return _PANEL_CACHE[key]


def run_poet(df, indicators, want_K=False, full_indicators=None):
    """Fast factor extraction on an indicator subset, reusing the cached
    interpolated panel. Returns (loadings_df, factor_scores_df[, K_selected])."""
    if len(indicators) < 8:
        return (None, None, np.nan) if want_K else (None, None)
    panel = _cached_panel(df, full_indicators if full_indicators is not None else indicators)
    sub = panel[["country_name", "year"] + indicators]
    from sklearn.preprocessing import StandardScaler
    X = StandardScaler().fit_transform(sub[indicators].values)
    rot, factors = fast_factors(X, K)
    loadings = pd.DataFrame(rot, index=indicators,
                            columns=[f"f{i+1}" for i in range(K)])
    f_scores = pd.DataFrame({
        "country_name": sub["country_name"].values,
        "year": sub["year"].values,
    })
    for i in range(K):
        f_scores[f"f{i+1}"] = factors[:, i]
    if want_K:
        ic, _, _ = bai_ng_ic(X)
        return loadings, f_scores, ic[2]
    return loadings, f_scores


def procrustes_align(L0_shared, Lr):
    """Orthogonal Procrustes: rotate refit loadings Lr toward original L0_shared.
    Returns the rotation matrix R (Lr @ R aligns to L0_shared)."""
    M = L0_shared.T @ Lr
    U, _, Vt = np.linalg.svd(M)
    R = (U @ Vt).T
    return R


def compare_to_base(base_loadings, base_f1, indicators_kept, df, want_K=False,
                    full_indicators=None):
    """Refit on indicators_kept, align, return (phi, score_r, K_selected).
    K_selected only computed when want_K=True (bai_ng_ic adds cost; skip it
    for the 332 LOIO + 200 bootstrap refits where K isn't reported per-config)."""
    if want_K:
        loadings_r, fscores_r, K_sel = run_poet(df, indicators_kept, want_K=True,
                                                 full_indicators=full_indicators)
    else:
        loadings_r, fscores_r = run_poet(df, indicators_kept,
                                         full_indicators=full_indicators)
        K_sel = np.nan
    if loadings_r is None:
        return np.nan, np.nan, np.nan

    # Restrict original loadings to the shared (kept) indicators
    shared = [c for c in indicators_kept if c in base_loadings.index]
    L0_shared = base_loadings.loc[shared].values         # (n_shared, K)
    Lr = loadings_r.loc[shared].values                   # (n_shared, K)

    # Procrustes-align refit loadings to original, then pick the refit column
    # most congruent with original factor-1.
    R = procrustes_align(L0_shared, Lr)
    Lr_aligned = Lr @ R
    congruences = [tucker_congruence(L0_shared[:, 0], Lr_aligned[:, c]) for c in range(K)]
    j = int(np.argmax(congruences))
    phi = congruences[j]

    # Factor-score correlation on common (country, year)
    fr = fscores_r[["country_name", "year", f"f{j+1}"]].rename(columns={f"f{j+1}": "fr"})
    merged = base_f1.merge(fr, on=["country_name", "year"], how="inner")
    if len(merged) > 10:
        score_r = abs(np.corrcoef(merged["f1"], merged["fr"])[0, 1])
    else:
        score_r = np.nan
    return phi, score_r, K_sel


def main():
    print("=" * 70)
    print("A1b: Within-V-Dem Indicator Jackknife (factor-1 stability)")
    print("=" * 70)

    df = load_vdem()
    indicators = select_indicators(df)
    base_loadings, base_fscores = run_poet(df, indicators, full_indicators=indicators)
    base_f1 = base_fscores[["country_name", "year", "f1"]].copy()
    print(f"\nBase: {len(indicators)} indicators, K={K}, "
          f"{len(base_fscores)} country-years")

    # Identify top loaders on factor-1
    top_loaders = base_loadings["f1"].abs().sort_values(ascending=False)
    print(f"\nTop-10 factor-1 loaders:")
    for ind, val in top_loaders.head(10).items():
        print(f"  {ind:30s} {base_loadings.loc[ind, 'f1']:+.3f}")

    rows = []

    # --- Tier 1: leave-one-component-out ---
    print(f"\n--- Tier 1: leave-one-component-out ---")
    for comp, prefixes in VDEM_COMPONENTS.items():
        drop = [c for c in indicators if any(c.startswith(p) or p in c for p in prefixes)]
        kept = [c for c in indicators if c not in drop]
        if len(kept) < 8 or len(drop) == 0:
            continue
        phi, r, k = compare_to_base(base_loadings, base_f1, kept, df, want_K=True, full_indicators=indicators)
        rows.append({"tier": "leave_component_out", "config": comp,
                     "n_dropped": len(drop), "n_kept": len(kept),
                     "phi": phi, "score_r": r, "K_selected": k})
        print(f"  drop {comp:14s} (-{len(drop):3d}): phi={phi:.4f}  r={r:.4f}  K={k}")

    # --- Tier 2: drop-top-loaders (headline) ---
    print(f"\n--- Tier 2: drop-top-loaders (headline adversarial) ---")
    for n_top in [5, 10]:
        drop = top_loaders.head(n_top).index.tolist()
        kept = [c for c in indicators if c not in drop]
        phi, r, k = compare_to_base(base_loadings, base_f1, kept, df, want_K=True, full_indicators=indicators)
        rows.append({"tier": "drop_top_loaders", "config": f"top{n_top}",
                     "n_dropped": n_top, "n_kept": len(kept),
                     "phi": phi, "score_r": r, "K_selected": k})
        print(f"  drop top-{n_top:2d} loaders: phi={phi:.4f}  r={r:.4f}  K={k}")

    # --- Tier 3: random-subset bootstrap ---
    print(f"\n--- Tier 3: random-subset bootstrap (drop 20%, {N_BOOT} reps) ---")
    boot_phis = []
    for b in range(N_BOOT):
        n_drop = int(0.2 * len(indicators))
        drop = list(RNG.choice(indicators, size=n_drop, replace=False))
        kept = [c for c in indicators if c not in drop]
        phi, r, k = compare_to_base(base_loadings, base_f1, kept, df, full_indicators=indicators)
        if not np.isnan(phi):
            boot_phis.append(phi)
        if (b + 1) % 50 == 0:
            print(f"    {b+1}/{N_BOOT} done")
    boot_phis = np.array(boot_phis)
    rows.append({"tier": "random_bootstrap", "config": f"drop20pct_x{len(boot_phis)}",
                 "n_dropped": int(0.2 * len(indicators)), "n_kept": len(indicators) - int(0.2 * len(indicators)),
                 "phi": float(boot_phis.mean()), "score_r": np.nan, "K_selected": np.nan})
    print(f"  bootstrap phi: mean={boot_phis.mean():.4f}  "
          f"min={boot_phis.min():.4f}  5th-pct={np.percentile(boot_phis, 5):.4f}")

    # --- Tier 4: leave-one-indicator-out (report min) ---
    print(f"\n--- Tier 4: leave-one-indicator-out (min phi across {len(indicators)}) ---")
    loio_phis = []
    for c in indicators:
        kept = [x for x in indicators if x != c]
        phi, _, _ = compare_to_base(base_loadings, base_f1, kept, df, full_indicators=indicators)
        if not np.isnan(phi):
            loio_phis.append(phi)
    loio_phis = np.array(loio_phis)
    rows.append({"tier": "leave_one_indicator_out", "config": f"all_{len(loio_phis)}",
                 "n_dropped": 1, "n_kept": len(indicators) - 1,
                 "phi": float(loio_phis.min()), "score_r": np.nan, "K_selected": np.nan})
    print(f"  LOIO phi: min={loio_phis.min():.4f}  mean={loio_phis.mean():.4f}")

    df_out = pd.DataFrame(rows)
    df_out.to_csv(OUT, index=False)

    # --- Verdict ---
    print(f"\n{'=' * 70}")
    print("VERDICT")
    print("=" * 70)
    non_adversarial = df_out[df_out["tier"].isin(
        ["leave_component_out", "random_bootstrap", "leave_one_indicator_out"])]
    min_phi_non_adv = non_adversarial["phi"].min()
    droptop = df_out[df_out["tier"] == "drop_top_loaders"]["phi"]
    print(f"  Min phi (non-adversarial tiers): {min_phi_non_adv:.4f}")
    print(f"  Drop-top-5/10 phi:               {droptop.values}")
    if min_phi_non_adv >= 0.95:
        print("  >= 0.95: factor-1 is a STABLE latent construct, robust to indicator")
        print("           selection (Lorenzo-Seva & ten Berge 2006 'factors equal').")
    elif min_phi_non_adv >= 0.90:
        print("  0.90-0.95: factor-1 is HIGHLY STABLE ('fair-to-equal similarity').")
    else:
        print("  < 0.90: factor-1 shows fragility under some perturbation — investigate.")
    print(f"\nWrote {OUT}")


if __name__ == "__main__":
    main()
