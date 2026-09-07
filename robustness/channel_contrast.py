"""Paired contrast between the mobilization and digital-control blocks under the strict design.

onset_forecast_clean.py and onset_forecast_ert.py report each block's marginal contribution
(full minus ablated) with a paired country-clustered bootstrap. Those intervals bound one
block at a time. The contrast between the two blocks is a different quantity: the difference
between the two contributions, which on the same scored rows equals AUC(ablate_dsp) minus
AUC(ablate_mob), because the full-model term cancels. This script re-runs the identical
rolling scorers for full, ablate_mob and ablate_dsp and bootstraps that difference directly,
pooling over the same three learners and three seeds, then reports the minimum detectable
contrast at eighty percent power (2.80 * SE with SE = width / 3.92).

AIM4D_OUTCOME=ledger uses the hand ledger (onset_forecast_clean); AIM4D_OUTCOME=ert uses
ERT v16 (onset_forecast_ert). Outputs robustness/channel_contrast_<outcome>.csv.
"""

import os
import sys

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

OUT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, OUT)
OUTCOME = os.environ.get("AIM4D_OUTCOME", "ledger")


def main():
    if OUTCOME == "ert":
        import onset_forecast_ert as M
        d = M.build_panel_ert()
        onsets = d.attrs["onsets"]
        drop = ("v2x_regime", "onset_year", "ep_end", "in_episode", "at_risk", "n_episodes")
    else:
        import onset_forecast_clean as M
        d = M.build_panel()
        drop = ("v2x_regime", "onset_year", "ep_end", "in_episode", "at_risk")
    feats = [c for c in d.columns if c not in M.EXCLUDE_COLS and c not in drop and d[c].dtype != object]
    mob = [c for c in feats if M.is_mob(c)]
    dsp = [c for c in feats if M.is_dsp(c)]
    for h in M.HORIZONS:
        if OUTCOME == "ert":
            d.attrs["onsets"] = onsets
            d[f"y{h}"] = M.label_ert(d, h, future_only=True)
        else:
            d[f"y{h}"] = M.label_h(d, h)
    print(f"{OUTCOME}: features {len(feats)} | mobilization {len(mob)} | dsp {len(dsp)}", flush=True)

    rows = []
    for h in M.HORIZONS:
        for name in ["gb", "rf", "lr"]:
            for seed in M.SEEDS:
                got = {}
                for cfg, fl in {"full": feats,
                                "ablate_mob": [c for c in feats if c not in set(mob)],
                                "ablate_dsp": [c for c in feats if c not in set(dsp)]}.items():
                    r = M.rolling_scores(d, fl, h, name, seed)
                    if r is not None:
                        got[cfg] = r
                if {"full", "ablate_mob", "ablate_dsp"} > set(got):
                    continue
                y, pf, c, _ = got["full"]
                _, pm, _, _ = got["ablate_mob"]
                _, pd_, _, _ = got["ablate_dsp"]
                (ma, la, ha, pa), _ = M.paired_boot(y, pd_, pm, c)
                rows.append({"h": h, "learner": name, "seed": seed, "n_scored": len(y), "n_pos": int(y.sum()),
                             "contrib_mob": round(roc_auc_score(y, pf) - roc_auc_score(y, pm), 4),
                             "contrib_dsp": round(roc_auc_score(y, pf) - roc_auc_score(y, pd_), 4),
                             "contrast_mean": round(ma, 4), "contrast_lo": round(la, 4),
                             "contrast_hi": round(ha, 4), "contrast_p_gt0": round(pa, 3)})
                print(f"h={h} {name} s{seed} mob-minus-dsp {ma:+.4f} [{la:+.4f},{ha:+.4f}]", flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(OUT, f"channel_contrast_{OUTCOME}.csv"), index=False)
    g = df.groupby("h")[["contrib_mob", "contrib_dsp", "contrast_mean", "contrast_lo", "contrast_hi"]].mean()
    g["mde_80"] = 2.80 * (g.contrast_hi - g.contrast_lo) / 3.92
    print(f"\n=== {OUTCOME}: pooled over learners and seeds (mobilization minus digital control) ===")
    print(g.round(4).to_string())
    print(f"\nWrote channel_contrast_{OUTCOME}.csv")


if __name__ == "__main__":
    main()
