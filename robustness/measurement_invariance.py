"""
A1: Measurement-invariance check — rebuild the Stage-1 democratic factor on
Freedom House + Polity indicators (instead of V-Dem) and show factor-1
correlates with the V-Dem factor-1.

Per the research: at ~31 indicators use PCA + parallel analysis, NOT POET
(POET's thresholding consistency is large-P asymptotics). The clean test
window is 2013-2018 (FH 25-subquestions start 2013; Polity5 ends 2018).

Go/no-go: Pearson >= 0.80 on the overlapping country-years. Expected 0.85-0.90
per the convergent-validity literature (Coppedge et al. 2018: V-Dem vs FH ~0.90,
vs Polity2 ~0.85; Boese 2019).

Output: robustness/measurement_invariance.csv + stdout verdict.
"""

import os
import sys
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from scipy.stats import pearsonr, spearmanr

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

DATA = os.path.join(REPO, "data")
FH_CSV = os.path.join(DATA, "fh_subscores.csv")
POL_CSV = os.path.join(DATA, "polity5.csv")
COW_MAP = os.path.join(DATA, "cow_iso3_mapping.csv")
VDEM_FACTORS = os.path.join(REPO, "stage1_factors", "country_year_factors.csv")
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "measurement_invariance.csv")

OVERLAP_START, OVERLAP_END = 2013, 2018


def _parallel_analysis(X, n_iter=100, percentile=95, rng=None):
    """Horn's parallel analysis: keep factors whose eigenvalue exceeds the
    95th-percentile eigenvalue from random data of the same shape."""
    rng = rng or np.random.default_rng(0)
    n, p = X.shape
    real_eigs = np.sort(np.linalg.eigvalsh(np.corrcoef(X, rowvar=False)))[::-1]
    rand_eigs = np.zeros((n_iter, p))
    for i in range(n_iter):
        Xr = rng.standard_normal((n, p))
        rand_eigs[i] = np.sort(np.linalg.eigvalsh(np.corrcoef(Xr, rowvar=False)))[::-1]
    thresh = np.percentile(rand_eigs, percentile, axis=0)
    k = int(np.sum(real_eigs > thresh))
    return max(k, 1), real_eigs, thresh


def _fh_country_to_iso3(name):
    """Best-effort FH country name -> V-Dem country_text_id via country_converter."""
    try:
        import country_converter as coco
        iso = coco.convert(names=name, to="ISO3", not_found=None)
        return iso
    except Exception:
        return None


def load_fh():
    if not os.path.exists(FH_CSV):
        sys.exit(f"Missing {FH_CSV}. Run: python3 data/download_fh_polity.py")
    fh = pd.read_csv(FH_CSV)
    subq = [c for c in fh.columns if c not in ("country", "year")]
    fh = fh.dropna(subset=subq, how="all")
    # map country -> iso3
    try:
        import country_converter as coco
        cc = coco.CountryConverter()
        mapped = cc.convert(names=fh["country"].tolist(), to="ISO3", not_found=None)
    except ImportError:
        sys.exit("Need country_converter: pip install country_converter")
    # coco returns a LIST for ambiguous multi-match names and None for
    # unmatched — both are unusable as merge keys. Keep only clean strings.
    fh["country_text_id"] = [m if isinstance(m, str) else None for m in mapped]
    fh = fh.dropna(subset=["country_text_id"])
    return fh, subq


def load_polity():
    if not os.path.exists(POL_CSV):
        sys.exit(f"Missing {POL_CSV}. Run: python3 data/download_fh_polity.py")
    pol = pd.read_csv(POL_CSV)
    comps = [c for c in ["xrcomp", "xropen", "xconst", "parreg", "parcomp"]
             if c in pol.columns]
    # map ccode (COW) -> country_text_id
    cow = pd.read_csv(COW_MAP).set_index("COWcode")["country_text_id"]
    pol["country_text_id"] = pol["ccode"].map(cow)
    pol = pol.dropna(subset=["country_text_id"])
    # Polity uses -66/-77/-88 as special codes; treat as missing
    for c in comps:
        pol[c] = pol[c].where(pol[c] >= -10, np.nan)
    return pol, comps


def main():
    print("=" * 70)
    print("A1: Measurement Invariance — FH+Polity factor vs V-Dem factor-1")
    print("=" * 70)

    fh, fh_cols = load_fh()
    pol, pol_cols = load_polity()
    print(f"FH: {len(fh)} rows, {len(fh_cols)} subquestions")
    print(f"Polity5: {len(pol)} rows, {len(pol_cols)} components")

    # Merge FH + Polity on (country_text_id, year), restrict to overlap window
    merged = fh.merge(pol, on=["country_text_id", "year"], how="inner",
                      suffixes=("_fh", "_pol"))
    merged = merged[(merged["year"] >= OVERLAP_START) & (merged["year"] <= OVERLAP_END)]
    indicators = fh_cols + pol_cols
    merged = merged.dropna(subset=indicators)
    print(f"Triple-overlap ({OVERLAP_START}-{OVERLAP_END}): {len(merged)} country-years, "
          f"{len(indicators)} indicators")

    if len(merged) < 100:
        sys.exit("Too few overlapping country-years for a stable factor model.")

    # Standardize + PCA
    X = StandardScaler().fit_transform(merged[indicators].values)
    k_pa, real_eigs, thresh = _parallel_analysis(X)
    print(f"Parallel analysis suggests K={k_pa} "
          f"(eig1={real_eigs[0]:.2f} vs rand-thresh {thresh[0]:.2f})")

    pca = PCA(n_components=max(k_pa, 1))
    scores = pca.fit_transform(X)
    fh_polity_f1 = scores[:, 0]
    # sign-align so higher = more democratic (use a free-expression-like FH item,
    # FH scores: higher = more free; D1 = media freedom)
    ref = merged["D1"].values if "D1" in merged.columns else merged[indicators[0]].values
    if np.corrcoef(fh_polity_f1, ref)[0, 1] < 0:
        fh_polity_f1 = -fh_polity_f1

    merged = merged.copy()
    merged["fh_polity_f1"] = fh_polity_f1

    # Load V-Dem factor-1 and merge
    vdem = pd.read_csv(VDEM_FACTORS)[["country_text_id", "year", "factor_1"]]
    cmp = merged[["country_text_id", "year", "fh_polity_f1"]].merge(
        vdem, on=["country_text_id", "year"], how="inner")
    print(f"\nComparison sample (FH+Polity ∩ V-Dem): {len(cmp)} country-years")

    if len(cmp) < 50:
        sys.exit("Too few rows to compare to V-Dem factor-1.")

    # Sign-align to V-Dem factor-1
    if np.corrcoef(cmp["fh_polity_f1"], cmp["factor_1"])[0, 1] < 0:
        cmp["fh_polity_f1"] = -cmp["fh_polity_f1"]

    r, _ = pearsonr(cmp["fh_polity_f1"], cmp["factor_1"])
    rho, _ = spearmanr(cmp["fh_polity_f1"], cmp["factor_1"])

    rows = [{
        "metric": "factor1_correlation",
        "pearson": r, "spearman": rho,
        "n": len(cmp), "k_parallel_analysis": k_pa,
        "n_indicators": len(indicators),
        "window": f"{OVERLAP_START}-{OVERLAP_END}",
        "var_explained_pc1": float(pca.explained_variance_ratio_[0]),
    }]
    pd.DataFrame(rows).to_csv(OUT, index=False)

    print(f"\n{'=' * 70}")
    print("RESULT")
    print("=" * 70)
    print(f"  Factor-1 (FH+Polity PCA) vs Factor-1 (V-Dem POET):")
    print(f"    Pearson  r = {r:.4f}")
    print(f"    Spearman ρ = {rho:.4f}")
    print(f"    PC1 explains {100*pca.explained_variance_ratio_[0]:.1f}% of FH+Polity variance")
    print()
    if r >= 0.85:
        print("  >= 0.85: clean MEASUREMENT-AGNOSTIC claim. The latent democratic")
        print("           factor reproduces on FH+Polity inputs; framework does not")
        print("           depend on V-Dem-specific indicators.")
    elif r >= 0.80:
        print("  0.80-0.85: DEFENSIBLE. Factor-1 transfers across measurement")
        print("           traditions (r≈0.8), consistent with the 0.85-0.90 aggregate")
        print("           convergent-validity literature (Coppedge 2018, Boese 2019).")
    else:
        print(f"  < 0.80 (r={r:.3f}): do NOT claim full independence. Fall back to the")
        print("           within-V-Dem jackknife (robustness/factor_jackknife.py) as the")
        print("           primary measurement-robustness evidence. Check sign/rotation")
        print("           alignment and whether FH's bundled subquestions collapse to")
        print("           fewer factors than V-Dem.")
    print(f"\nWrote {OUT}")


if __name__ == "__main__":
    main()
