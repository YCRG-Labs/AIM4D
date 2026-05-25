"""Does the staged representation earn its keep? The clean, non-circular test
(Koh et al. 2020 concept-bottleneck logic; Bernanke-Boivin-Eliasz 2005 FAVAR):
build the regime classifier ON the extracted factors versus ON the raw 332
indicators, and ask which agrees better with the EXTERNAL Regimes-of-the-World
typology that neither was fit on. If the factor representation yields
better-validated regimes, the upstream factor stage demonstrably improves the
downstream stage rather than merely sitting beside it.

Non-circular because RoW is external (used for neither fit) and the factors are
extracted unsupervised. Reuses the simplified HMM + kappa-vs-RoW machinery from
sanity_checks.
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
from sklearn.preprocessing import StandardScaler

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from sanity_checks import build_sequences, fit_and_decode, kappa_vs_row
from external_benchmarks import load_indicator_panel
from baseline_comparison import build_labels

OUTPUT_DIR = os.path.dirname(os.path.abspath(__file__))


def raw_sequences():
    panel, indicators = load_indicator_panel()
    panel = panel.sort_values(["country_name", "year"])
    X = StandardScaler().fit_transform(panel[indicators].values)
    panel = panel.reset_index(drop=True)
    seqs, lengths, order, years = [], [], [], []
    for country, idx in panel.groupby("country_name").groups.items():
        idx = list(idx)
        if len(idx) < 10:
            continue
        seqs.append(X[idx])
        lengths.append(len(idx))
        order.append(country)
        years.append(panel.loc[idx, "year"].values)
    return np.concatenate(seqs), lengths, order, years


def main():
    print("=" * 70)
    print("REPRESENTATION VALUE: regime recovery on factors vs raw indicators")
    print("=" * 70)

    Xf, lf, of_, yf = build_sequences()
    fac_states = fit_and_decode(Xf, lf, of_, yf)
    k_fac = kappa_vs_row(fac_states)
    print(f"\n  regime HMM on 4 extracted factors : weighted kappa vs RoW = {k_fac:.3f}  "
          f"(dim={Xf.shape[1]})")

    Xr, lr, or_, yr = raw_sequences()
    raw_states = fit_and_decode(Xr, lr, or_, yr)
    k_raw = kappa_vs_row(raw_states)
    print(f"  regime HMM on 332 raw indicators  : weighted kappa vs RoW = {k_raw:.3f}  "
          f"(dim={Xr.shape[1]})")

    print(f"\n  => factor representation {'IMPROVES' if k_fac > k_raw else 'does not improve'} "
          f"downstream regime recovery (delta = {k_fac - k_raw:+.3f})")
    pd.DataFrame([{"kappa_factors": k_fac, "kappa_raw": k_raw, "delta": k_fac - k_raw,
                   "dim_factors": Xf.shape[1], "dim_raw": Xr.shape[1]}]).to_csv(
        os.path.join(OUTPUT_DIR, "representation_value_results.csv"), index=False)
    print(f"\nSaved to robustness/representation_value_results.csv")


if __name__ == "__main__":
    main()
