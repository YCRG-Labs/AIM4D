"""
End-to-end strict forecasting: rebuild the whole pipeline at each forecast origin.

The strict evaluation elsewhere in this paper closes the downstream learner's
training windows but leaves the upstream representation fit once on the whole
panel, so the factors, betas, regime states and network components entering the
forecaster were constructed with knowledge of years after the origin. This
removes that. At origin T all five stages are refit on data through T, which is
everything a forecaster would hold at T, the downstream learner is trained only
on rows whose outcome window closed by T-h, and the rows dated T are scored.

Comparing the result against the full-sample-representation figure bounds how
much the shared representation was worth.

Outputs robustness/strict_endtoend_refit.csv.
"""

import os
import shutil
import sys
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from _refit_worktree import (add_worktree, default_workers, refit_env,
                             remove_worktree, run_stages)

OUT = os.path.dirname(os.path.abspath(__file__))
REPO_DATA_ROOT = os.path.join(os.path.dirname(OUT), "data")
H = 5
ORIGINS = [int(x) for x in os.environ.get("AIM4D_ORIGINS", "2012,2014,2016,2018,2020").split(",")]
EXCLUDE = {"country_name", "country_text_id", "year", "label", "label_soft",
           "combined_risk", "calibrated_risk", "alert_tier", "combined_alert",
           "combined_alert_legacy", "ews_alert", "raw_alert", "election_alert",
           "dem_vulnerability_alert", "military_threat_alert", "mv_csd_alert",
           "n_factors", "is_postonset", "onset_year", "peak_year", "ep_end",
           "in_episode", "at_risk", "v2x_regime", "y"}


def panel_from(worktree):
    from stage5_ews.estimate import KNOWN_EPISODES
    d = pd.read_csv(os.path.join(worktree, "stage5_ews", "ews_signals.csv"))
    v = pd.read_csv(os.path.join(worktree, "data", "vdem_v16.csv"), low_memory=False,
                    usecols=["country_name", "year", "v2x_regime"]).dropna(subset=["v2x_regime"])
    v["v2x_regime"] = v["v2x_regime"].astype(int)
    d = d.merge(v, on=["country_name", "year"], how="left")
    d["onset_year"] = d["country_name"].map({c: i["onset"] for c, i in KNOWN_EPISODES.items()})
    d["peak_year"] = d["country_name"].map({c: i.get("peak", i["onset"]) for c, i in KNOWN_EPISODES.items()})
    span = (d.onset_year.notna() & (d.year >= d.onset_year) & (d.year <= d.peak_year))
    ip = d["is_postonset"].fillna(False).astype(bool) if "is_postonset" in d.columns else False
    d["in_episode"] = span | ip
    d["at_risk"] = (d.v2x_regime >= 2) & (~d.in_episode)
    fut = d.onset_year - d.year
    d["y"] = ((fut >= 1) & (fut <= H)).astype(int)
    return d


def truncated_data_dir(T):
    """A data directory holding nothing dated after T.

    Setting AIM4D_CUTOFF restricts what each stage fits on, but the worktrees
    symlink the canonical data directory, so the filters and decoders still run
    across the whole panel: a smoothed regime state at year t is computed from a
    sequence that extends to 2025. Truncating the inputs removes that, because
    no observation after T exists anywhere in the run.
    """
    dst = os.path.join(REPO_DATA_ROOT, f"_trunc_{T}")
    if os.path.isdir(dst):
        shutil.rmtree(dst)
    os.makedirs(dst)
    src = os.path.join(os.path.dirname(OUT), "data")
    for name in os.listdir(src):
        if name.startswith(("_trunc_", "__pycache__")):
            continue
        sp, dp = os.path.join(src, name), os.path.join(dst, name)
        if name.endswith(".csv"):
            try:
                df = pd.read_csv(sp, low_memory=False)
            except Exception:
                os.symlink(sp, dp); continue
            if "year" in df.columns:
                df = df[pd.to_numeric(df["year"], errors="coerce") <= T]
            df.to_csv(dp, index=False)
        else:
            os.symlink(sp, dp)
    return dst


def score_origin(T):
    t0 = time.time()
    wt = add_worktree(f"e2e_{T}")
    try:
        tdir = truncated_data_dir(T)
        link = os.path.join(wt, "data")
        if os.path.islink(link) or os.path.exists(link):
            os.remove(link) if os.path.islink(link) else shutil.rmtree(link)
        os.symlink(tdir, link)
        env = refit_env(AIM4D_CUTOFF=T, AIM4D_EXCLUDE_COUNTRY=None)
        rc = run_stages(wt, env, label=f" origin={T}")
        if rc != 0:
            print(f"origin {T}: stages failed rc={rc}", flush=True)
            return None
        d = panel_from(wt)
        feats = [c for c in d.columns if c not in EXCLUDE and d[c].dtype != object]
        tr = d[(d.year <= T - H) & d.at_risk]
        te = d[(d.year == T) & d.at_risk]
        if len(te) == 0 or tr.y.sum() < 5:
            print(f"origin {T}: insufficient rows", flush=True)
            return None
        sc = StandardScaler()
        X = sc.fit_transform(tr[feats].fillna(0).values)
        Xt = sc.transform(te[feats].fillna(0).values)
        out = []
        for name, mk in [("gb", lambda: GradientBoostingClassifier(
                              n_estimators=200, max_depth=3, learning_rate=0.05,
                              subsample=0.8, min_samples_leaf=10, random_state=0)),
                         ("rf", lambda: RandomForestClassifier(
                              n_estimators=400, min_samples_leaf=3, class_weight="balanced",
                              random_state=0, n_jobs=-1)),
                         ("lr", lambda: LogisticRegression(
                              penalty="elasticnet", solver="saga", l1_ratio=0.5, C=0.1,
                              max_iter=3000, class_weight="balanced"))]:
            m = mk()
            m.fit(X, tr.y.values)
            out.append(pd.DataFrame({"origin": T, "learner": name,
                                     "country_name": te.country_name.values,
                                     "y": te.y.values, "p": m.predict_proba(Xt)[:, 1]}))
        print(f"origin {T}: done in {(time.time()-t0)/60:.1f} min, "
              f"{len(te)} scored, {int(te.y.sum())} positive, {len(feats)} features", flush=True)
        return pd.concat(out, ignore_index=True)
    finally:
        remove_worktree(wt)
        td = os.path.join(REPO_DATA_ROOT, f"_trunc_{T}")
        if os.path.isdir(td):
            shutil.rmtree(td, ignore_errors=True)


def main():
    workers = int(os.environ.get("AIM4D_PAR", default_workers()))
    print(f"origins {ORIGINS}, {workers} workers, refitting all five stages per origin\n", flush=True)
    with ThreadPoolExecutor(max_workers=workers) as ex:
        res = [r for r in ex.map(score_origin, ORIGINS) if r is not None]
    if not res:
        print("no origins completed"); return
    d = pd.concat(res, ignore_index=True)
    d.to_csv(os.path.join(OUT, "strict_endtoend_refit.csv"), index=False)
    print("\n=== end-to-end refit, pooled across the completed origins ===")
    for name, g in d.groupby("learner"):
        if g.y.nunique() > 1:
            print(f"  {name}: AUC {roc_auc_score(g.y, g.p):.3f}  "
                  f"AP {average_precision_score(g.y, g.p):.3f}  "
                  f"n={len(g)} pos={int(g.y.sum())} base={g.y.mean():.3f}")
    from scipy.stats import rankdata
    b = d.pivot_table(index=["origin", "country_name", "y"], columns="learner", values="p").reset_index()
    cols = [c for c in ["gb", "rf", "lr"] if c in b.columns]
    # Rank within each origin, not across the pooled evaluation rows.
    def within(v):
        r = np.zeros(len(v))
        for o in np.unique(b["origin"].values):
            sel = b["origin"].values == o
            r[sel] = rankdata(v[sel]) / sel.sum()
        return r
    b["blend"] = np.mean([within(b[c].values) for c in cols], axis=0)
    print(f"  rank-mean blend: AUC {roc_auc_score(b.y, b.blend):.3f}  "
          f"AP {average_precision_score(b.y, b.blend):.3f}  n={len(b)} pos={int(b.y.sum())}")
    print("\nWrote strict_endtoend_refit.csv")


if __name__ == "__main__":
    main()
