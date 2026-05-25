"""Stage-3 regime-model robustness.

  lr_test       : parametric-bootstrap likelihood-ratio test for K=5 vs K=4 states
                  (Qu-Shi-Shum logic). Standard asymptotics fail for the number of
                  regimes, so the null distribution of LR = 2(LL5 - LL4) is built by
                  simulating from the fitted K=4 model and refitting both K on each
                  draw.
  confusion     : confusion matrix of the production states against V-Dem Regimes
                  of the World, beyond the aggregate weighted kappa.
  sojourn       : empirical mean state durations (run lengths) from the production
                  state sequences, checked against the persistence the typology
                  implies.
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
from hmmlearn import hmm
from sklearn.metrics import confusion_matrix, cohen_kappa_score

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from sanity_checks import build_sequences

OUTPUT_DIR = os.path.dirname(os.path.abspath(__file__))
N_BOOT = 100
RESTARTS = 3
STATE_NAMES = ["lib_dem", "elec_dem", "hybrid", "comp_auth", "clos_auth"]


def best_hmm(X, lengths, k, restarts=RESTARTS):
    best, bll = None, -np.inf
    for r in range(restarts):
        m = hmm.GaussianHMM(n_components=k, covariance_type="diag", min_covar=0.05,
                            n_iter=100, tol=1e-4, random_state=r)
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                m.fit(X, lengths)
                ll = m.score(X, lengths)
            if ll > bll:
                bll, best = ll, m
        except Exception:
            continue
    return best, bll


def sample_under(model, lengths):
    return np.concatenate([model.sample(L)[0] for L in lengths])


def lr_test(X, lengths):
    m4, ll4 = best_hmm(X, lengths, 4)
    m5, ll5 = best_hmm(X, lengths, 5)
    lr_obs = 2 * (ll5 - ll4)
    null = []
    for b in range(N_BOOT):
        Xb = sample_under(m4, lengths)
        _, l4 = best_hmm(Xb, lengths, 4)
        _, l5 = best_hmm(Xb, lengths, 5)
        null.append(2 * (l5 - l4))
    null = np.array(null)
    p = (np.sum(null >= lr_obs) + 1) / (N_BOOT + 1)
    return lr_obs, p, null


def states_vs_row():
    sdf = pd.read_csv(os.path.join(REPO, "stage3_msvar", "country_year_states.csv"))
    vdem = pd.read_csv(os.path.join(REPO, "data", "vdem_v16.csv"), low_memory=False,
                       usecols=["country_name", "year", "v2x_regime", "v2x_polyarchy"])
    vdem = vdem.dropna(subset=["v2x_regime"])
    vdem["v2x_regime"] = vdem["v2x_regime"].astype(int)

    def to5(reg, poly):
        if reg == 3: return 0
        if reg == 2: return 1
        if reg == 1: return 2 if (poly is not None and poly > 0.35) else 3
        return 4

    m = sdf.merge(vdem, on=["country_name", "year"], how="inner")
    m["ref"] = [to5(r, p) for r, p in zip(m["v2x_regime"], m["v2x_polyarchy"])]
    cm = confusion_matrix(m["ref"], m["state"], labels=list(range(5)))
    kappa = cohen_kappa_score(m["ref"], m["state"], weights="linear")
    return cm, kappa, sdf


def sojourn(sdf):
    runs = {s: [] for s in range(5)}
    for _, g in sdf.groupby("country_name"):
        st = g.sort_values("year")["state"].values
        if len(st) == 0:
            continue
        cur, length = st[0], 1
        for s in st[1:]:
            if s == cur:
                length += 1
            else:
                runs[cur].append(length)
                cur, length = s, 1
        runs[cur].append(length)
    return {s: (np.mean(v) if v else np.nan) for s, v in runs.items()}


def main():
    print("=" * 70)
    print("STAGE 3 REGIME-MODEL ROBUSTNESS")
    print("=" * 70)
    out = {}

    print("\n[A] Parametric-bootstrap LR test for K=5 vs K=4")
    X, lengths, order, years = build_sequences()
    lr_obs, p, null = lr_test(X, lengths)
    print(f"    observed LR = {lr_obs:.1f}; bootstrap null mean {null.mean():.1f} (sd {null.std():.1f})")
    print(f"    bootstrap p = {p:.3f}  ({'K=5 preferred' if p < 0.05 else 'K=5 not supported over K=4'})")
    out.update({"lr_obs": lr_obs, "lr_boot_p": p})

    print("\n[B] Confusion matrix vs V-Dem Regimes of the World (weighted kappa)")
    cm, kappa, sdf = states_vs_row()
    print(f"    weighted kappa = {kappa:.3f}")
    print("    rows = V-Dem RoW, cols = our state")
    print(pd.DataFrame(cm, index=STATE_NAMES, columns=STATE_NAMES).to_string())
    out["kappa_w"] = kappa

    print("\n[C] Empirical mean sojourn times (run length, years)")
    soj = sojourn(sdf)
    for s in range(5):
        print(f"    {STATE_NAMES[s]:10s}: {soj[s]:.1f}")
        out[f"sojourn_{STATE_NAMES[s]}"] = soj[s]

    pd.DataFrame([out]).to_csv(os.path.join(OUTPUT_DIR, "hmm_robustness_results.csv"), index=False)
    print(f"\nSaved to robustness/hmm_robustness_results.csv")


if __name__ == "__main__":
    main()
