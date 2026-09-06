"""
Onset forecasting on a clean risk set, with rolling origins and clustered inference.

This addresses three referee objections at once.

Risk set (objection 1). The primary target is onset among countries that are
democratic and not already inside an episode. Following the two-stage logic of
Boese et al. (2021), the at-risk pool at time t is country-years with V-Dem
Regimes of the World >= 2 (electoral or liberal democracy) and no ongoing
autocratization episode. Episodes beginning from an autocracy are a different
outcome (ERT's "regressed autocracy") and are excluded from the onset target
rather than pooled into it.

Information set (objection 2). Y_it^(h) = 1 if a country at risk at t
experiences an onset during t+1..t+h. Every predictor is dated t or earlier, so
the design is genuinely h-step-ahead rather than contemporaneous. Forecasts roll:
at origin T the model trains only on rows whose full h-year label window closed
at or before T, then scores the rows dated T.

Uncertainty (objection 5). Ablation differences are reported as paired
country-clustered bootstrap intervals on the same scored rows, across several
learners and seeds, rather than as a single number from one common fit.

ERT VARIANT. Identical to onset_forecast_clean.py except that the outcome comes
from the unmodified ERT v16 release via ert_panel, so recurrent onsets are
labelled and episode membership is the union of all ERT intervals. The feature
matrix, learners, seeds, horizons and bootstrap are unchanged.

Outputs robustness/onset_forecast_ert.csv and onset_ablation_ci_ert.csv.
"""

import os
import sys
import warnings

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from ert_panel import build_panel_ert, label_ert

OUT = os.path.dirname(os.path.abspath(__file__))
HORIZONS = [2, 5]
LAST_OBS = 2025
# An origin is admissible only when its whole outcome window falls inside the
# observed panel. With onsets observed through 2025 a later origin would score
# rows against onsets that cannot yet have been recorded, counting them as
# non-events. Origins are therefore horizon-specific.
ORIGINS = list(range(2005, LAST_OBS + 1))


def origins_for(h):
    return [t for t in ORIGINS if t <= LAST_OBS - h]
N_BOOT = 2000
SEEDS = [0, 1, 2]
RNG = np.random.default_rng(20260904)

DSP_PREFIXES = ("v2smgovdom", "v2smfordom", "v2smgovfilprc", "v2smgovsmmon", "v2smpardom")
EXCLUDE_COLS = {
    "country_name", "country_text_id", "year", "label", "label_soft",
    "combined_risk", "calibrated_risk", "alert_tier",
    "combined_alert", "combined_alert_legacy", "ews_alert", "raw_alert",
    "election_alert", "dem_vulnerability_alert", "military_threat_alert",
    "mv_csd_alert", "n_factors", "is_postonset",
    # Episode bookkeeping. These describe the outcome and must never reach a
    # model: onset_year and peak_year are non-null only for countries that have
    # an episode, and their values say when it happened. Leaving peak_year out
    # of this set once raised the elastic-net AUC from 0.64 to 0.85.
    "onset_year", "peak_year", "ep_end", "in_episode", "at_risk",
}


def is_dsp(c):
    return any(c.startswith(p) for p in DSP_PREFIXES)


def is_mob(c):
    return c.lower().startswith("v2ca")


def build_panel():
    from stage5_ews.estimate import KNOWN_EPISODES  # noqa: F401  (unused in ERT variant)
    d = pd.read_csv(os.path.join(OUT, "..", "stage5_ews", "ews_signals.csv"))
    v = pd.read_csv(os.path.join(OUT, "..", "data", "vdem_v16.csv"), low_memory=False,
                    usecols=["country_name", "year", "v2x_regime"])
    v = v.dropna(subset=["v2x_regime"])
    v["v2x_regime"] = v["v2x_regime"].astype(int)
    d = d.merge(v, on=["country_name", "year"], how="left")

    onset = {c: info["onset"] for c, info in KNOWN_EPISODES.items()}
    peak = {c: info.get("peak", info["onset"]) for c, info in KNOWN_EPISODES.items()}
    d["onset_year"] = d["country_name"].map(onset)
    d["peak_year"] = d["country_name"].map(peak)

    # An episode runs from its onset to its peak. Treating a country as
    # permanently in-episode after any onset removes exactly the units most
    # exposed to a second one: Venezuela's 2002 onset would exclude it for the
    # following 23 years. Membership therefore ends at the peak, after which a
    # country that is still coded a democracy re-enters the at-risk pool.
    span = (d["onset_year"].notna()
            & (d["year"] >= d["onset_year"])
            & (d["year"] <= d["peak_year"]))
    in_ep = d["is_postonset"].fillna(False).astype(bool) if "is_postonset" in d.columns else False
    d["in_episode"] = span | in_ep
    d["at_risk"] = (d["v2x_regime"] >= 2) & (~d["in_episode"])
    return d


def label_h(d, h):
    fut = d["onset_year"] - d["year"]
    return ((fut >= 1) & (fut <= h)).astype(int)


def learner(name, seed):
    if name == "gb":
        return GradientBoostingClassifier(n_estimators=100, max_depth=3, learning_rate=0.05,
                                          subsample=0.8, min_samples_leaf=20, random_state=seed)
    if name == "rf":
        return RandomForestClassifier(n_estimators=300, min_samples_leaf=5,
                                      random_state=seed, n_jobs=-1)
    return LogisticRegression(C=1.0, max_iter=2000, random_state=seed)


def rolling_scores(d, feats, h, name, seed):
    y_all, p_all, c_all, yr_all = [], [], [], []
    for T in origins_for(h):
        tr = d[(d.year <= T - h) & d.at_risk]
        te = d[(d.year == T) & d.at_risk]
        if len(te) == 0 or tr[f"y{h}"].sum() < 5:
            continue
        sc = StandardScaler()
        Xtr = sc.fit_transform(tr[feats].fillna(0).values)
        Xte = sc.transform(te[feats].fillna(0).values)
        m = learner(name, seed)
        try:
            m.fit(Xtr, tr[f"y{h}"].values)
            p = m.predict_proba(Xte)[:, 1]
        except Exception:
            continue
        y_all.append(te[f"y{h}"].values); p_all.append(p)
        c_all.append(te["country_name"].values); yr_all.append(te["year"].values)
    if not y_all:
        return None
    return (np.concatenate(y_all), np.concatenate(p_all),
            np.concatenate(c_all), np.concatenate(yr_all))


def paired_boot(y, pa, pb, countries):
    uniq = np.unique(countries)
    idx_by_c = {c: np.where(countries == c)[0] for c in uniq}
    d_auc, d_ap = [], []
    for _ in range(N_BOOT):
        draw = RNG.choice(uniq, size=len(uniq), replace=True)
        idx = np.concatenate([idx_by_c[c] for c in draw])
        if y[idx].sum() < 3 or y[idx].sum() == len(idx):
            continue
        try:
            d_auc.append(roc_auc_score(y[idx], pa[idx]) - roc_auc_score(y[idx], pb[idx]))
            d_ap.append(average_precision_score(y[idx], pa[idx]) - average_precision_score(y[idx], pb[idx]))
        except Exception:
            continue
    f = lambda a: (float(np.mean(a)), float(np.percentile(a, 2.5)), float(np.percentile(a, 97.5)),
                   float(np.mean(np.array(a) > 0)))
    return f(d_auc), f(d_ap)


def main():
    d = build_panel_ert()
    _onsets = d.attrs['onsets']
    feats = [c for c in d.columns if c not in EXCLUDE_COLS
             and c not in ("v2x_regime", "onset_year", "ep_end", "in_episode", "at_risk", "n_episodes")
             and d[c].dtype != object]
    mob = [c for c in feats if is_mob(c)]
    dsp = [c for c in feats if is_dsp(c)]
    print(f"features {len(feats)} | mobilization {len(mob)} | dsp {len(dsp)}")

    ar = d[d.at_risk]
    print(f"at-risk country-years (democracy, not in episode): {len(ar)} of {len(d)}")
    for h in HORIZONS:
        d.attrs["onsets"] = _onsets
        d[f"y{h}"] = label_ert(d, h, future_only=True)
        n_pos = int(d.loc[d.at_risk, f"y{h}"].sum())
        print(f"  h={h}: {n_pos} positive at-risk country-years "
              f"({n_pos/len(ar)*100:.1f}% base rate), "
              f"{d.loc[d.at_risk & (d[f'y{h}']==1), 'country_name'].nunique()} distinct countries")
    print()

    rows, ci_rows = [], []
    for h in HORIZONS:
        for name in ["gb", "rf", "lr"]:
            for seed in SEEDS:
                cfgs = {"full": feats,
                        "ablate_mob": [c for c in feats if c not in set(mob)],
                        "ablate_dsp": [c for c in feats if c not in set(dsp)]}
                got = {}
                for cfg, fl in cfgs.items():
                    r = rolling_scores(d, fl, h, name, seed)
                    if r is None:
                        continue
                    y, p, c, yr = r
                    got[cfg] = (y, p, c)
                    rows.append({"h": h, "learner": name, "seed": seed, "config": cfg,
                                 "n_scored": len(y), "n_pos": int(y.sum()),
                                 "auc_roc": roc_auc_score(y, p) if y.sum() else np.nan,
                                 "auc_pr": average_precision_score(y, p) if y.sum() else np.nan})
                if {"full", "ablate_mob", "ablate_dsp"} <= set(got):
                    y, pf, c = got["full"]
                    for blk in ["mob", "dsp"]:
                        _, pab, _ = got[f"ablate_{blk}"]
                        (ma, la, ha, pa_), (mp, lp, hp, pp_) = paired_boot(y, pf, pab, c)
                        ci_rows.append({"h": h, "learner": name, "seed": seed, "block": blk,
                                        "d_auc_mean": round(ma, 4), "d_auc_lo": round(la, 4),
                                        "d_auc_hi": round(ha, 4), "d_auc_p_gt0": round(pa_, 3),
                                        "d_ap_mean": round(mp, 4), "d_ap_lo": round(lp, 4),
                                        "d_ap_hi": round(hp, 4), "d_ap_p_gt0": round(pp_, 3)})
                        print(f"h={h} {name} s{seed} {blk:<4} "
                              f"dAUC {ma:+.4f} [{la:+.4f},{ha:+.4f}]  "
                              f"dAP {mp:+.4f} [{lp:+.4f},{hp:+.4f}]")

    pd.DataFrame(rows).to_csv(os.path.join(OUT, "onset_forecast_ert.csv"), index=False)
    ci = pd.DataFrame(ci_rows)
    ci.to_csv(os.path.join(OUT, "onset_ablation_ci_ert.csv"), index=False)
    print("\n=== pooled across learners and seeds (full minus ablated; positive = block helps) ===")
    print(ci.groupby(["h", "block"])[["d_auc_mean", "d_auc_lo", "d_auc_hi",
                                      "d_ap_mean", "d_ap_lo", "d_ap_hi"]].mean().round(4).to_string())
    print("\nWrote onset_forecast_ert.csv and onset_ablation_ci_ert.csv")


if __name__ == "__main__":
    main()
