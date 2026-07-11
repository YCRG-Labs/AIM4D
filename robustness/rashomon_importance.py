"""Rashomon variable-importance for the headline finding (mobilization > digital
information control). Permutation importance from a single model can be a one-model
artifact; the Fisher-Rudin-Dominici model-class-reliance / Rashomon-set idea is to
check whether the ranking holds across ALL near-optimal models. We fit a spread of
candidate models, keep those within epsilon of the best out-of-sample AUC (the
Rashomon set), and report channel-level permutation importance across that set. The
claim survives iff the mobilization channel outranks the digital-society (DSP)
channel in essentially every near-optimal model.
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
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier, ExtraTreesClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUTPUT_DIR = os.path.dirname(os.path.abspath(__file__))
RNG = np.random.default_rng(42)
EPS = 0.03
N_PERM = 10


def channel_of(c):
    c = c.lower()
    if c.startswith("v2ca"):
        return "mobilization"
    if c.startswith("v2sm"):
        return "digital control"
    if any(k in c for k in ["csd", "var_z", "ar1_z", "kurt_z", "skew_z", "dom_eig", "xcorr"]):
        return "critical slowing down"
    if "network" in c or "contagion" in c:
        return "network exposure"
    if "election" in c or "party_threat" in c:
        return "election vulnerability"
    if "mil_" in c:
        return "military threat"
    if c.startswith("v2exl"):
        return "executive legitimation"
    if c.startswith("factor_") or c.startswith("f1_") or "_c22_" in c:
        return "latent factor dynamics"
    return "structural / macro"


def main():
    print("=" * 70)
    print("RASHOMON VARIABLE IMPORTANCE: mobilization vs digital control")
    print("=" * 70)
    e = pd.read_csv(os.path.join(REPO, "stage5_ews", "ews_signals.csv"))
    drop = {"country_name", "country_text_id", "year", "label", "is_postonset",
            "combined_risk", "calibrated_risk", "combined_alert", "alert_tier",
            "ews_alert", "raw_alert", "mv_csd_alert", "election_alert",
            "dem_vulnerability_alert", "military_threat_alert", "combined_alert_legacy", "label_soft"}
    feats = [c for c in e.columns if c not in drop and e[c].dtype != object]
    chan = {f: channel_of(f) for f in feats}
    channels = sorted(set(chan.values()))

    tr = e[e["year"] <= 2019]
    te = e[(e["year"] > 2019) & (~e["is_postonset"].astype(bool))]
    sc = StandardScaler().fit(tr[feats].fillna(0).values)
    Xtr, ytr = sc.transform(tr[feats].fillna(0).values), tr["label"].values
    Xte, yte = sc.transform(te[feats].fillna(0).values), te["label"].values

    candidates = []
    for C in [0.1, 0.5, 1.0, 2.0]:
        candidates.append(LogisticRegression(C=C, max_iter=2000, class_weight="balanced", random_state=0))
    for d in [2, 3, 4]:
        for lr in [0.05, 0.1]:
            candidates.append(HistGradientBoostingClassifier(max_depth=d, learning_rate=lr, max_iter=200, random_state=0))
    candidates.append(RandomForestClassifier(n_estimators=300, max_depth=6, class_weight="balanced", random_state=0, n_jobs=4))
    candidates.append(ExtraTreesClassifier(n_estimators=300, max_depth=6, class_weight="balanced", random_state=0, n_jobs=4))

    fitted = []
    for m in candidates:
        m.fit(Xtr, ytr)
        auc = roc_auc_score(yte, m.predict_proba(Xte)[:, 1])
        fitted.append((m, auc))
    best = max(a for _, a in fitted)
    rashomon = [(m, a) for m, a in fitted if a >= best - EPS]
    print(f"\n  best OOS AUC = {best:.3f}; Rashomon set (within {EPS}) = {len(rashomon)}/{len(fitted)} models")

    col_idx = {ch: [i for i, f in enumerate(feats) if chan[f] == ch] for ch in channels}
    rows = []
    mob_gt_dsp = 0
    for m, a in rashomon:
        base = roc_auc_score(yte, m.predict_proba(Xte)[:, 1])
        imp = {}
        for ch, idx in col_idx.items():
            if not idx:
                continue
            drops = []
            for _ in range(N_PERM):
                Xp = Xte.copy()
                perm = RNG.permutation(len(Xp))
                Xp[:, idx] = Xp[perm][:, idx]
                drops.append(base - roc_auc_score(yte, m.predict_proba(Xp)[:, 1]))
            imp[ch] = float(np.mean(drops))
        rows.append(imp)
        if imp.get("mobilization", 0) > imp.get("digital control", 0):
            mob_gt_dsp += 1

    imp_df = pd.DataFrame(rows)
    print("\n  channel permutation importance across the Rashomon set (OOS AUC drop):")
    summary = imp_df.agg(["mean", "min", "max"]).T.sort_values("mean", ascending=False)
    print(summary.to_string(float_format="%.4f"))
    print(f"\n  mobilization > digital control in {mob_gt_dsp}/{len(rashomon)} near-optimal models")
    print(f"  mobilization importance range [{imp_df['mobilization'].min():.4f}, {imp_df['mobilization'].max():.4f}]; "
          f"digital control range [{imp_df['digital control'].min():.4f}, {imp_df['digital control'].max():.4f}]")
    summary.to_csv(os.path.join(OUTPUT_DIR, "rashomon_importance_results.csv"))
    imp_df.to_csv(os.path.join(OUTPUT_DIR, "rashomon_importance_per_model.csv"), index=False)
    print(f"\nSaved to robustness/rashomon_importance_results.csv")


if __name__ == "__main__":
    main()
