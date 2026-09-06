"""
Numeric audit of the manuscript's passport claims against pipeline outputs.

For each claim in quality_reports/passports/aim4d-fullpaper.yaml this recomputes
the value from the named output file and compares it against the value the
manuscript reports, at the tolerance the passport entry declares. It also
confirms the reported figure actually appears in the .tex, which catches the
case where an output and a passport entry agree but the paper was never updated.

Read-only with respect to the pipeline: it opens result CSVs and the manuscript
and writes nothing but its own report.
"""

import os
import re
import sys
import unicodedata

import numpy as np
import pandas as pd

REPO = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
ROB = os.path.join(REPO, "robustness")
TEX = os.environ.get("AIM4D_TEX", os.path.expanduser("~/aim4d-paper/fullpaper.tex"))

TEXT = open(TEX, encoding="utf-8").read()

rows = []


def check(cid, desc, reported, computed, tol=0.005, in_tex=None, kind="point"):
    if kind == "exact":
        ok = reported == computed
        diff = "exact" if ok else f"{reported} vs {computed}"
    else:
        diff = abs(reported - computed)
        ok = diff <= tol
        diff = f"{diff:.4f}"
    tex_ok = True
    if in_tex is not None:
        tex_ok = any(s in TEXT for s in ([in_tex] if isinstance(in_tex, str) else in_tex))
    rows.append({
        "claim": cid, "what": desc, "reported": reported, "computed": computed,
        "diff": diff, "tol": tol if kind != "exact" else "exact",
        "num": "PASS" if ok else "FAIL",
        "in_tex": "yes" if tex_ok else "NOT FOUND",
    })


def csv(name, sub=ROB):
    return pd.read_csv(os.path.join(sub, name))


cv = csv("expanding_window_cv.csv")
check("C1", "refit CV mean AUC", 0.821, round(cv["auc"].mean(), 3), in_tex="0.821")
check("C1", "refit CV mean AP", 0.524, round(cv["ap"].mean(), 3), in_tex="0.524")

b = csv("bootstrap_cis.csv").set_index("metric")
for cid, metric, rep, lo, hi, tx in [
    ("C2", "auc_roc_oos_2019", 0.939, 0.800, 0.975, "0.939"),
    ("C2", "auc_pr_oos_2019", 0.591, 0.354, 0.759, "0.591"),
    ("C2", "bss_oos_2019", 0.212, -0.003, 0.329, "0.212"),
    ("C7", "auc_fh_3yr_decline_2pt", 0.767, 0.720, 0.831, "0.77"),
    ("C7", "auc_polity_3yr_decline_3pt", 0.736, 0.689, 0.774, "0.74"),
]:
    if metric in b.index:
        r = b.loc[metric]
        check(cid, f"{metric} point", rep, round(float(r["point"]), 3), in_tex=tx)
        check(cid, f"{metric} CI lo", lo, round(float(r["ci_low"]), 3), tol=0.01)
        check(cid, f"{metric} CI hi", hi, round(float(r["ci_high"]), 3), tol=0.01)
    else:
        rows.append({"claim": cid, "what": metric, "reported": rep, "computed": "-",
                     "diff": "-", "tol": "-", "num": "UNMATCHED", "in_tex": "-"})

lt = csv("lead_time_auc.csv")
lt = lt[lt["lead_years"].astype(str).str.fullmatch(r"\d+")].copy()
lt["lead_years"] = lt["lead_years"].astype(int)
lt = lt.set_index("lead_years")
for lead, rep in [(1, 0.956), (2, 0.903), (3, 0.854), (4, 0.835)]:
    check("C3", f"lead {lead} AUC", rep, round(float(lt.loc[lead, "auc_roc"]), 3), in_tex=str(rep))

lo = csv("loeo_results.csv", sub=os.path.join(REPO, "stage5_ews"))
w = int(lo["detected_watch"].sum())
wn = int(lo["detected_warning"].sum())
al = int(lo["detected_alert"].sum())
check("C4", "LOEO watch", 35, w, kind="exact", in_tex="35/46")
check("C4", "LOEO warning", 22, wn, kind="exact", in_tex="22/46")
check("C4", "LOEO alert", 12, al, kind="exact", in_tex="12/46")
if "type" in lo.columns:
    bs = lo[lo["type"].astype(str).str.contains("backslid", case=False, na=False)]
    cp = lo[~lo.index.isin(bs.index)]
    check("C4", "backsliding detected", 24, int(bs["detected_watch"].sum()), kind="exact", in_tex="24/32")
    check("C4", "coup detected", 11, int(cp["detected_watch"].sum()), kind="exact", in_tex="11/14")

sp = csv("sample_pipeline_loeo.csv")
check("C5", "15-ep LOEO mean delta", 0.081, round(float(sp["delta_risk"].mean()), 3), in_tex="0.081")
check("C5", "15-ep higher count", 14, int((sp["delta_risk"] > 0).sum()), kind="exact",
      in_tex=["fourteen of the fifteen", "fourteen of fifteen"])

bf = csv("benchmark_finishers_results.csv")
r0 = bf.iloc[0]
check("C6", "booster AUC raw", 0.940, round(float(r0["auc_full"]), 3), in_tex="0.940")
check("C6", "booster AUC-PR raw", 0.735, round(float(r0["aucpr_full"]), 3), in_tex="0.735")
check("C6", "booster AUC-PR clean", 0.634, round(float(r0["aucpr_clean"]), 3), in_tex=["0.634", "0.63"])

pi = csv("permutation_importance_oos.csv").set_index("feature")
for feat, rep in [("f1_rolling_mean", 0.019), ("f1_change", 0.013), ("v2cademmob", 0.010),
                  ("v2cagenmob_detrended", 0.009), ("v2smgovdom", 0.008)]:
    check("C8", f"perm imp {feat}", rep, round(float(pi.loc[feat, "mean_delta_auc"]), 3), tol=0.001)

rz = csv("rashomon_importance_per_model.csv")
check("C9", "mobilization mean imp", 0.069, round(float(rz["mobilization"].mean()), 3), in_tex="0.069")
check("C9", "digital control mean imp", 0.063, round(float(rz["digital control"].mean()), 3), in_tex="0.063")
check("C9", "mob > dsp rows", 21, int((rz["mobilization"] > rz["digital control"]).sum()),
      kind="exact", in_tex="21 of the 30")
check("C9", "mob > factor rows", 0, int((rz["mobilization"] > rz["latent factor dynamics"]).sum()),
      kind="exact")

mp = csv("mobilization_precedence_results.csv").iloc[0]
check("C10", "mob z at t-5", 0.19, round(float(mp["mob_tau-5"]), 2), tol=0.01, in_tex="0.19")
check("C10", "mob z at t-1", 0.47, round(float(mp["mob_tau-1"]), 2), tol=0.01, in_tex="0.47")
for col, rep, tx in [("mob_fires", 25, "25 of the 48"), ("dig_fires", 8, "8 for digital control")]:
    if col in mp.index:
        check("C10", col, rep, int(mp[col]), kind="exact", in_tex=tx)

da = csv("dsp_ablation.csv").set_index("configuration")
check("C11", "DSP full OOS", 0.905, round(float(da.loc["full", "auc_roc_oos_2017"]), 3), in_tex="0.905")
check("C11", "DSP ablated OOS", 0.899, round(float(da.loc["ablate_dsp", "auc_roc_oos_2017"]), 3), in_tex="0.899")
check("C11", "DSP only OOS", 0.774, round(float(da.loc["dsp_only", "auc_roc_oos_2017"]), 3), in_tex="0.774")

sa = csv("stage_ablation_results.csv")
g = sa.groupby("ablate_group")[["oos_roc", "oos_pr"]].mean()
if "full" in g.index and "drop_stage4" in g.index:
    check("C12", "stage4 full ROC", 0.935, round(float(g.loc["full", "oos_roc"]), 3), in_tex="0.935")
    check("C12", "stage4 drop ROC", 0.927, round(float(g.loc["drop_stage4", "oos_roc"]), 3), in_tex="0.927")
    check("C12", "stage4 full PR", 0.661, round(float(g.loc["full", "oos_pr"]), 3), in_tex="0.661")
    check("C12", "stage4 drop PR", 0.686, round(float(g.loc["drop_stage4", "oos_pr"]), 3), in_tex="0.686")

ns = csv("network_seed_sweep_summary.csv").set_index(csv("network_seed_sweep_summary.csv").columns[0])
for col, rep in [("alpha_contig", 0.24), ("alpha_alliance", 0.26), ("alpha_trade", 0.22),
                 ("alpha_cultural", 0.28)]:
    check("C13", f"sweep {col}", rep, round(float(ns.loc["mean", col]), 2), tol=0.01)

cs = csv("contagion_seed_sweep_summary.csv").set_index("country")
for c, rep, sd, tx in [("Hungary", 0.606, 0.064, "60"), ("Turkey", 0.323, 0.037, "32")]:
    def _norm(x):
        return unicodedata.normalize("NFKD", str(x)).encode("ascii", "ignore").decode().lower()
    aliases = {"Turkey": ("turkiye", "turkey")}.get(c, (c.lower(),))
    key = next((k for k in cs.index if any(a in _norm(k) for a in aliases)), None)
    if key:
        check("C14", f"{c} contagion mean", rep, round(float(cs.loc[key, "mean"]), 3))
        check("C14", f"{c} contagion sd", sd, round(float(cs.loc[key, "std"]), 3), tol=0.01)
    else:
        rows.append({"claim": "C14", "what": c, "reported": rep, "computed": "-", "diff": "-",
                     "tol": "-", "num": "UNMATCHED", "in_tex": "-"})

vu = csv("vdem_uncertainty_results.csv").iloc[0]
check("C15", "vdem unc AUC", 0.92, round(float(vu["auc_mean"]), 2), tol=0.01, in_tex="0.92")
check("C15", "vdem unc sd", 0.010, round(float(vu["auc_sd"]), 3), in_tex="0.010")
check("C15", "mob>dsp draws", 29, int(vu["mob_gt_dsp"]), kind="exact", in_tex="twenty-nine of thirty")

en = csv("elastic_net_robustness.csv")
row_en = en[en["n_features_enet_selected"].notna()].iloc[0]
check("C16", "enet selected", 89, int(row_en["n_features_enet_selected"]), kind="exact", in_tex="89")
check("C16", "enet OOS AUC", 0.932, round(float(row_en["oos_auc"]), 3), in_tex="0.932")
check("C16", "enet OOS AUC-PR", 0.602, round(float(row_en["oos_auc_pr"]), 3), in_tex="0.602")

sc = csv("sanity_checks_results.csv").iloc[0]
check("C17", "perm congruence", 0.06, round(float(sc["factor_perm_congruence_mean"]), 2), tol=0.01, in_tex="0.06")
check("C17", "hmm kappa real", 0.58, round(float(sc["hmm_kappa_real"]), 2), tol=0.01, in_tex="0.58")
check("C17", "hmm kappa scrambled", -0.01, round(float(sc["hmm_kappa_scrambled"]), 2), tol=0.01)

# The baseline ladder is now scored on the rows common to every model
# (robustness/baseline_common_rows.py), so the manuscript carries those figures
# rather than this standalone file's. Both are checked: the standalone against
# its own output, the common-row version against the table the paper prints.
mo = csv("mobilization_only_baseline.csv").iloc[0]
check("C19", "mob-only AUC (standalone)", 0.716, round(float(mo["auc_roc"]), 3))
check("C19", "mob-only AUC-PR (standalone)", 0.151, round(float(mo["auc_pr"]), 3))
bc = csv("baseline_common_rows.csv").set_index("model")
for _m, _roc, _pr in [
    ("Five-stage framework (AIM4D)", 0.939, 0.591),
    ("Persistence (3-yr polyarchy decline)", 0.826, 0.324),
    ("Mobilization-only logit", 0.756, 0.163),
    ("Elastic net, V-Dem indicators", 0.813, 0.254),
    ("Gradient boosting, V-Dem indicators", 0.923, 0.639),
    ("V-Forecast ensemble", 0.891, 0.493),
]:
    check("C19b", f"{_m} AUC", _roc, round(float(bc.loc[_m, "auc_roc"]), 3), in_tex=f"{_roc:.3f}")
    check("C19b", f"{_m} AUC-PR", _pr, round(float(bc.loc[_m, "auc_pr"]), 3), in_tex=f"{_pr:.3f}")

rb = csv("reliability_bins.csv")
for i, (mpred, obs, n) in enumerate([(0.091, 0.009, 464), (0.183, 0.044, 203), (0.309, 0.500, 40),
                                     (0.423, 0.696, 23), (0.549, 0.833, 6)]):
    check("C20", f"reliability bin {i+1} pred", mpred, round(float(rb.iloc[i]["mean_predicted"]), 3))
    check("C20", f"reliability bin {i+1} obs", obs, round(float(rb.iloc[i]["observed_freq"]), 3))
    check("C20", f"reliability bin {i+1} n", n, int(rb.iloc[i]["n"]), kind="exact")

ca = csv("channel_ablation.csv").set_index("configuration")
full_oos = float(ca.loc["full", "auc_roc_oos"])
for blk, rep_auc, rep_pr, rep_only in [("dsp", -0.007, -0.010, 0.776), ("mob", -0.042, -0.097, 0.774),
                                       ("factor", -0.033, -0.146, 0.843)]:
    d_auc = float(ca.loc[f"ablate_{blk}", "auc_roc_oos"]) - full_oos
    d_pr = float(ca.loc[f"ablate_{blk}", "auc_pr_oos"]) - float(ca.loc["full", "auc_pr_oos"])
    check("C21", f"{blk} d_AUC", rep_auc, round(d_auc, 3))
    check("C21", f"{blk} d_AUC_PR", rep_pr, round(d_pr, 3))
    check("C21", f"{blk}_only OOS", rep_only, round(float(ca.loc[f"{blk}_only", "auc_roc_oos"]), 3))

for blk, rep_auc, rep_pr, rep_only in [("mobdem", -0.009, -0.057, 0.714), ("mobaut", -0.021, -0.065, 0.631),
                                       ("mobgen", -0.002, -0.033, 0.644)]:
    d_auc = float(ca.loc[f"ablate_{blk}", "auc_roc_oos"]) - full_oos
    d_pr = float(ca.loc[f"ablate_{blk}", "auc_pr_oos"]) - float(ca.loc["full", "auc_pr_oos"])
    check("C24", f"{blk} d_AUC", rep_auc, round(d_auc, 3))
    check("C24", f"{blk} d_AUC_PR", rep_pr, round(d_pr, 3))
    check("C24", f"{blk}_only OOS", rep_only, round(float(ca.loc[f"{blk}_only", "auc_roc_oos"]), 3))

ks = csv("k_sensitivity_results.csv")
check("C22", "locked kappa (K=4)", 0.72, round(float(ks[ks["K"] == 4]["weighted_kappa"].iloc[0]), 2),
      tol=0.01, in_tex="0.72")
hl = csv("hmm_states_locked_results.csv").set_index("S")
for S, kw, ku in [(3, 0.574, 0.444), (4, 0.647, 0.479), (5, 0.718, 0.500), (6, 0.699, 0.387)]:
    check("C22", f"locked sweep S={S} kappa_w", kw, round(float(hl.loc[S, "kappa_w"]), 3), in_tex=f"{kw:.3f}")
    check("C22", f"locked sweep S={S} kappa", ku, round(float(hl.loc[S, "kappa"]), 3), in_tex=f"{ku:.3f}")
check("C22", "locked S=5 reproduces pipeline", 0.72, round(float(hl.loc[5, "kappa_w"]), 2), tol=0.01)
check("C22", "sanity kappa", 0.58, round(float(sc["hmm_kappa_real"]), 2), tol=0.01, in_tex="0.58")

out = pd.DataFrame(rows)
n_fail = int((out["num"] == "FAIL").sum())
n_unm = int((out["num"] == "UNMATCHED").sum())
n_tex = int((out["in_tex"] == "NOT FOUND").sum())

pd.set_option("display.width", 200)
pd.set_option("display.max_rows", 400)
print(out.to_string(index=False))
print(f"\nPASS {int((out['num']=='PASS').sum())}   FAIL {n_fail}   "
      f"UNMATCHED {n_unm}   value-absent-from-tex {n_tex}   total {len(out)}")

if n_fail:
    print("\nFAILING ROWS")
    print(out[out["num"] == "FAIL"].to_string(index=False))
if n_tex:
    print("\nREPORTED VALUE NOT FOUND IN MANUSCRIPT")
    print(out[out["in_tex"] == "NOT FOUND"].to_string(index=False))

sys.exit(1 if n_fail else 0)
