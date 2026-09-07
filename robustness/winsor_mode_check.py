"""Symmetric versus upper-tail winsorization of the Stage 2 betas, on the current pipeline.

Appendix B states that switching from upper-tail to symmetric winsorization changes a
small number of beta observations and leaves the regime classifier's weighted kappa,
the 2019 hold-out AUC and the LOEO tier counts essentially unchanged. This re-derives
those four quantities by refitting the whole pipeline twice in isolated worktrees, once
per AIM4D_WINSOR_MODE, so the canonical outputs are never touched. Kappa is read from
the Stage 3 log of each run; the hold-out and LOEO figures are scored from each run's
artifacts with the same evaluate_fold used elsewhere.

Output: robustness/winsor_mode_check.csv (one row per mode) and a printed comparison.
"""

import os
import re
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _refit_worktree import STAGES, add_worktree, refit_env, remove_worktree
from expanding_window_cv import evaluate_fold

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "winsor_mode_check.csv")
LOGDIR = os.environ.get("AIM4D_WINSOR_LOGDIR", os.path.dirname(os.path.abspath(__file__)))


def run_all(worktree, env, log):
    with open(log, "w") as f:
        for st in STAGES:
            rc = subprocess.call([sys.executable, os.path.join(worktree, st)], env=env, cwd=worktree,
                                 stdout=f, stderr=subprocess.STDOUT)
            if rc != 0:
                return rc
    return 0


def one_mode(mode):
    wt = add_worktree(f"winsor_{mode}")
    log = os.path.join(LOGDIR, f"winsor_{mode}.log")
    try:
        env = refit_env(AIM4D_CUTOFF=2019, AIM4D_EXCLUDE_COUNTRY=None, AIM4D_WINSOR_MODE=mode)
        rc = run_all(wt, env, log)
        if rc != 0:
            raise SystemExit(f"{mode}: pipeline failed rc={rc}, see {log}")
        r = evaluate_fold(2019, 2026, repo=wt)
        loeo = pd.read_csv(os.path.join(wt, "stage5_ews", "loeo_results.csv"))
        kap = re.findall(r"Weighted kappa \(linear\): ([0-9.]+)", open(log).read())
        betas = pd.read_csv(os.path.join(wt, "stage2_betas", "country_year_betas.csv"))
        return {"mode": mode, "oos_auc": r["auc"], "oos_auc_pr": r["ap"], "n_pos": r["n_pos"],
                "kappa_w": float(kap[-1]) if kap else np.nan,
                "loeo_watch": int(loeo["tier"].isin(["watch", "warning", "alert"]).sum()),
                "loeo_warning": int(loeo["tier"].isin(["warning", "alert"]).sum()),
                "loeo_alert": int((loeo["tier"] == "alert").sum()), "loeo_total": len(loeo)}, betas
    finally:
        remove_worktree(wt)


def main():
    with ThreadPoolExecutor(2) as ex:
        res = list(ex.map(one_mode, ["symmetric", "upper"]))
    rows = [r for r, _ in res]
    b_sym, b_up = res[0][1], res[1][1]
    cols = [c for c in b_sym.columns if c.startswith("beta")]
    key = ["country_text_id", "year"] if "country_text_id" in b_sym.columns else ["country_name", "year"]
    m = b_sym.merge(b_up, on=key, suffixes=("_s", "_u"))
    diff = np.zeros(len(m), dtype=bool); total = 0; changed = 0
    for c in cols:
        d = (m[f"{c}_s"] - m[f"{c}_u"]).abs() > 1e-9
        changed += int(d.sum()); total += int(d.notna().sum())
    for r in rows:
        r["beta_cells_changed_vs_other"] = changed; r["beta_cells_total"] = total
    df = pd.DataFrame(rows); df.to_csv(OUT, index=False)
    print("\n=== winsorization mode check ===")
    print(df.round(4).to_string(index=False))
    print(f"\nbeta cells differing between modes: {changed} of {total}")
    print(f"Wrote {OUT}")


if __name__ == "__main__":
    main()
