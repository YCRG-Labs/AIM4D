"""Per-country risk-driver attribution for the prospective watchlist, replacing
the hand-labeled 'Primary Signals' column with a computed one.

Following the Early Warning Project's transparent approach (a regularized linear
attribution of the risk score: contribution_j = coef_j * x_ij, summed within
channels), we fit a linear surrogate of the meta-learner's risk score on the
standardized Stage-5 features, verify its fidelity, and report for each top-25
country the channel contributing most to its elevated risk. Channel-level
aggregation defeats the within-channel feature correlation that makes
single-feature attribution unreliable. The label is scoped to the model: it is
the signal that most raises a country's estimated risk, not a causal claim.
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
from sklearn.linear_model import RidgeCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUTPUT_DIR = os.path.dirname(os.path.abspath(__file__))


def channel_of(col):
    c = col.lower()
    if any(k in c for k in ["csd", "var_z", "ar1_z", "kurt_z", "skew_z", "dom_eig", "xcorr"]):
        return "critical slowing down"
    if "network" in c or "contagion" in c:
        return "network exposure"
    if "election" in c or "party_threat" in c:
        return "election vulnerability"
    if "mil_" in c:
        return "military threat"
    if c.startswith("v2ca"):
        return "mobilization"
    if c.startswith("v2sm"):
        return "digital information control"
    if c.startswith("v2exl"):
        return "executive legitimation"
    if c.startswith("factor_") or c.startswith("f1_") or "_c22_" in c:
        return "latent factor dynamics"
    return "structural / macro"


def main():
    print("=" * 70)
    print("PROSPECTIVE WATCHLIST: per-country risk drivers (linear attribution)")
    print("=" * 70)
    ews = pd.read_csv(os.path.join(REPO, "stage5_ews", "ews_signals.csv"))

    drop = {"country_name", "country_text_id", "year", "label", "is_postonset",
            "combined_risk", "calibrated_risk", "combined_alert", "alert_tier",
            "ews_alert", "raw_alert", "mv_csd_alert", "election_alert",
            "dem_vulnerability_alert", "military_threat_alert", "combined_alert_legacy",
            "label_soft"}
    feats = [c for c in ews.columns if c not in drop and ews[c].dtype != object]
    target = "combined_risk" if "combined_risk" in ews.columns else "calibrated_risk"

    d = ews.dropna(subset=[target]).copy()
    X = d[feats].fillna(0.0).values
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)
    eps = 1e-6
    y = np.log(np.clip(d[target].values, eps, 1 - eps) / np.clip(1 - d[target].values, eps, 1 - eps))

    surrogate = RidgeCV(alphas=np.logspace(-2, 3, 20)).fit(Xs, y)
    r2 = r2_score(y, surrogate.predict(Xs))
    print(f"\n  linear surrogate fidelity to risk-score logit: R^2 = {r2:.3f}  "
          f"({'faithful' if r2 > 0.8 else 'approximate, interpret with care'})")

    coef = surrogate.coef_
    contrib = Xs * coef[None, :]
    channels = np.array([channel_of(f) for f in feats])
    uniq = sorted(set(channels))

    latest = d.sort_values("year").groupby("country_name").tail(1)
    top = latest.sort_values(target, ascending=False).head(25)

    rows = []
    for _, r in top.iterrows():
        i = d.index.get_loc(r.name)
        row_contrib = contrib[i]
        ch_pos = {ch: row_contrib[channels == ch][row_contrib[channels == ch] > 0].sum() for ch in uniq}
        tot = sum(v for v in ch_pos.values() if v > 0)
        shares = {ch: (v / tot if tot > 0 else 0) for ch, v in ch_pos.items()}
        ranked = sorted(shares, key=shares.get, reverse=True)
        top1, top2 = ranked[0], ranked[1]
        gap = shares[top1] - shares[top2]
        primary = top1 if gap > 0.10 else f"mixed ({top1} / {top2})"
        rows.append({"country": r["country_name"], "year": int(r["year"]),
                     "risk": round(float(r[target]), 3), "primary_driver": primary,
                     "top_share": round(shares[top1], 2), "second": top2})

    out = pd.DataFrame(rows)
    print("\n  top-25 leading model signal:")
    print(out.to_string(index=False))
    out.to_csv(os.path.join(OUTPUT_DIR, "prospective_drivers_results.csv"), index=False)
    print(f"\n  channel frequency as primary driver:")
    print(out["primary_driver"].str.replace(r" \(.*\)", "", regex=True).value_counts().to_string())
    print(f"\nSaved to robustness/prospective_drivers_results.csv")


if __name__ == "__main__":
    main()
