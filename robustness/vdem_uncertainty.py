"""V-Dem measurement-uncertainty robustness (circularity / coding-artifact defense).

The Little-Meng critique is that expert-coded V-Dem signal could be a coding
artifact. V-Dem ships per-estimate measurement uncertainty (the _sd columns from
its Bayesian IRT model). We draw the raw V-Dem inputs from N(value, sd), refit the
Stage-5 forecaster, and check that the out-of-sample AUC and the mobilization >
digital-control ranking are stable across measurement-uncertainty draws. Stability
means the result reflects signal, not coding noise. (Scope: perturbs the raw V-Dem
features the meta-learner consumes directly; the derived channels are held fixed,
so this is a conservative lower bound on robustness.)
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
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUTPUT_DIR = os.path.dirname(os.path.abspath(__file__))
RNG = np.random.default_rng(42)
K_DRAWS = 30


def main():
    print("=" * 70)
    print("V-DEM MEASUREMENT-UNCERTAINTY ROBUSTNESS")
    print("=" * 70)
    ews = pd.read_csv(os.path.join(REPO, "stage5_ews", "ews_signals.csv"))
    drop = {"country_name", "country_text_id", "year", "label", "is_postonset",
            "combined_risk", "calibrated_risk", "combined_alert", "alert_tier",
            "ews_alert", "raw_alert", "mv_csd_alert", "election_alert",
            "dem_vulnerability_alert", "military_threat_alert", "combined_alert_legacy", "label_soft"}
    feats = [c for c in ews.columns if c not in drop and ews[c].dtype != object]

    raw_v2 = [c for c in feats if (c.startswith("v2ca") or c.startswith("v2sm") or c.startswith("v2exl"))
              and not c.endswith("_detrended") and "_x_post2015" not in c]
    sd_cols = [c + "_sd" for c in raw_v2]
    avail = pd.read_csv(os.path.join(REPO, "data", "vdem_v16.csv"), low_memory=False, nrows=1).columns
    sd_cols = [s for s in sd_cols if s in avail]
    perturb = [s[:-3] for s in sd_cols]
    vd = pd.read_csv(os.path.join(REPO, "data", "vdem_v16.csv"), low_memory=False,
                     usecols=["country_text_id", "year"] + sd_cols)
    ews = ews.merge(vd, on=["country_text_id", "year"], how="left")
    for s in sd_cols:
        ews[s] = ews[s].fillna(ews[s].median())
    print(f"  perturbing {len(perturb)} raw V-Dem features by their coding SD over {K_DRAWS} draws")

    mob = [c for c in feats if c.startswith("v2ca")]
    dsp = [c for c in feats if c.startswith("v2sm")]
    tr_mask = (ews["year"] <= 2019).values
    te_mask = ((ews["year"] > 2019) & (~ews["is_postonset"].astype(bool))).values
    y = ews["label"].values

    def run_once(Xfeat):
        sc = StandardScaler().fit(Xfeat[tr_mask])
        Xs = sc.transform(Xfeat)
        m = HistGradientBoostingClassifier(max_depth=3, learning_rate=0.05, max_iter=200, random_state=0)
        m.fit(Xs[tr_mask], y[tr_mask])
        base = roc_auc_score(y[te_mask], m.predict_proba(Xs[te_mask])[:, 1])

        def chimp(cols):
            idx = [feats.index(c) for c in cols]
            d = []
            for _ in range(5):
                Xp = Xs[te_mask].copy()
                Xp[:, idx] = Xp[RNG.permutation(len(Xp))][:, idx]
                d.append(base - roc_auc_score(y[te_mask], m.predict_proba(Xp)[:, 1]))
            return float(np.mean(d))
        return base, chimp(mob), chimp(dsp)

    base_vals = ews[feats].fillna(0.0).values.astype(float)
    sd_vals = ews[sd_cols].values
    pidx = [feats.index(p) for p in perturb]

    aucs, mob_gt = [], 0
    for k in range(K_DRAWS):
        X = base_vals.copy()
        X[:, pidx] = X[:, pidx] + RNG.normal(0, 1, X[:, pidx].shape) * sd_vals
        auc, im_mob, im_dsp = run_once(X)
        aucs.append(auc)
        mob_gt += int(im_mob > im_dsp)
    aucs = np.array(aucs)
    print(f"\n  OOS AUC across {K_DRAWS} measurement-uncertainty draws: "
          f"mean {aucs.mean():.3f}, sd {aucs.std():.3f}, range [{aucs.min():.3f}, {aucs.max():.3f}]")
    print(f"  mobilization > digital control in {mob_gt}/{K_DRAWS} draws")
    pd.DataFrame([{"auc_mean": aucs.mean(), "auc_sd": aucs.std(), "auc_min": aucs.min(),
                   "auc_max": aucs.max(), "mob_gt_dsp": mob_gt, "draws": K_DRAWS}]).to_csv(
        os.path.join(OUTPUT_DIR, "vdem_uncertainty_results.csv"), index=False)
    print(f"\nSaved to robustness/vdem_uncertainty_results.csv")


if __name__ == "__main__":
    main()
