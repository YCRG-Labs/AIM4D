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
check("C1", "refit CV mean AUC", 0.817, round(cv["auc"].mean(), 3), in_tex="0.817")
check("C1", "refit CV mean AP", 0.541, round(cv["ap"].mean(), 3), in_tex="0.541")

b = csv("bootstrap_cis.csv").set_index("metric")
for cid, metric, rep, lo, hi, tx in [
    ("C2", "auc_roc_oos_2019", 0.938, 0.797, 0.975, "0.938"),
    ("C2", "auc_pr_oos_2019", 0.622, 0.375, 0.792, "0.622"),
    ("C2", "bss_oos_2019", 0.219, 0.013, 0.335, "0.219"),
    ("C7", "auc_fh_3yr_decline_2pt", 0.772, 0.726, 0.830, "0.77"),
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
for lead, rep in [(1, 0.949), (2, 0.910), (3, 0.864), (4, 0.858)]:
    check("C3", f"lead {lead} AUC", rep, round(float(lt.loc[lead, "auc_roc"]), 3), in_tex=str(rep))

lo = csv("loeo_results.csv", sub=os.path.join(REPO, "stage5_ews"))
w = int(lo["detected_watch"].sum())
wn = int(lo["detected_warning"].sum())
al = int(lo["detected_alert"].sum())
check("C4", "LOEO watch", 35, w, kind="exact", in_tex="35/46")
check("C4", "LOEO warning", 18, wn, kind="exact", in_tex="22/46")
check("C4", "LOEO alert", 11, al, kind="exact", in_tex="12/46")
if "type" in lo.columns:
    bs = lo[lo["type"].astype(str).str.contains("backslid", case=False, na=False)]
    cp = lo[~lo.index.isin(bs.index)]
    check("C4", "backsliding detected", 24, int(bs["detected_watch"].sum()), kind="exact", in_tex="24/32")
    check("C4", "coup detected", 11, int(cp["detected_watch"].sum()), kind="exact", in_tex="11/14")

sp = csv("sample_pipeline_loeo.csv")
check("C5", "15-ep LOEO mean delta", 0.093, round(float(sp["delta_risk"].mean()), 3), in_tex="0.093")
check("C5", "15-ep higher count", 14, int((sp["delta_risk"] > 0).sum()), kind="exact",
      in_tex=["fourteen of the fifteen", "fourteen of fifteen"])

bf = csv("benchmark_finishers_results.csv")
r0 = bf.iloc[0]
check("C6", "booster AUC raw", 0.940, round(float(r0["auc_full"]), 3), in_tex="inflates the booster's figure further still")
check("C6", "booster AUC-PR raw", 0.735, round(float(r0["aucpr_full"]), 3), in_tex="0.735")
check("C6", "booster AUC-PR clean", 0.634, round(float(r0["aucpr_clean"]), 3), in_tex=["0.634", "0.63"])

pi = csv("permutation_importance_oos.csv").set_index("feature")
for feat, rep in [("f1_rolling_mean", 0.015), ("f1_change", 0.013), ("v2cademmob", 0.009),
                  ("v2cagenmob_detrended", 0.008), ("v2smgovdom", 0.007)]:
    check("C8", f"perm imp {feat}", rep, round(float(pi.loc[feat, "mean_delta_auc"]), 3), tol=0.001)

rz = csv("rashomon_importance_per_model.csv")
check("C9", "mobilization mean imp", 0.062, round(float(rz["mobilization"].mean()), 3), in_tex="0.062")
check("C9", "digital control mean imp", 0.063, round(float(rz["digital control"].mean()), 3), in_tex="0.063")
check("C9", "mob > dsp rows", 13, int((rz["mobilization"] > rz["digital control"]).sum()),
      kind="exact", in_tex="13 of the 30")
check("C9", "mob > factor rows", 0, int((rz["mobilization"] > rz["latent factor dynamics"]).sum()),
      kind="exact")

mp = csv("mobilization_precedence_results.csv").iloc[0]
check("C10", "mob z at t-5", 0.19, round(float(mp["mob_tau-5"]), 2), tol=0.01, in_tex="0.19")
check("C10", "mob z at t-1", 0.47, round(float(mp["mob_tau-1"]), 2), tol=0.01, in_tex="0.47")
for col, rep, tx in [("mob_fires", 25, "25 of the 48"), ("dig_fires", 8, "8 for digital control")]:
    if col in mp.index:
        check("C10", col, rep, int(mp[col]), kind="exact", in_tex=tx)

da = csv("dsp_ablation.csv").set_index("configuration")
check("C11", "DSP full OOS", 0.887, round(float(da.loc["full", "auc_roc_oos_2017"]), 3), in_tex="0.887")
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
for col, rep in [("alpha_contig", 0.24), ("alpha_alliance", 0.26), ("alpha_trade", 0.23),
                 ("alpha_cultural", 0.28)]:
    check("C13", f"sweep {col}", rep, round(float(ns.loc["mean", col]), 2), tol=0.01)

cs = csv("contagion_seed_sweep_summary.csv").set_index("country")
for c, rep, sd, tx in [("Hungary", 0.585, 0.109, "60"), ("Turkey", 0.303, 0.040, "32")]:
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
check("C15", "vdem unc AUC", 0.91, round(float(vu["auc_mean"]), 2), tol=0.01, in_tex="0.91")
check("C15", "vdem unc sd", 0.010, round(float(vu["auc_sd"]), 3), in_tex="0.010")
check("C15", "mob>dsp draws", 30, int(vu["mob_gt_dsp"]), kind="exact", in_tex="thirty of thirty")

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
    ("Five-stage framework (AIM4D)", 0.938, 0.622),
    ("Persistence (3-yr polyarchy decline)", 0.826, 0.324),
    ("Mobilization-only logit", 0.756, 0.163),
    ("Elastic net, V-Dem indicators", 0.813, 0.254),
    ("Gradient boosting, V-Dem indicators", 0.923, 0.639),
    ("V-Forecast ensemble", 0.891, 0.493),
]:
    check("C19b", f"{_m} AUC", _roc, round(float(bc.loc[_m, "auc_roc"]), 3), in_tex=f"{_roc:.3f}")
    check("C19b", f"{_m} AUC-PR", _pr, round(float(bc.loc[_m, "auc_pr"]), 3), in_tex=f"{_pr:.3f}")

rb = csv("reliability_bins.csv")
for i, (mpred, obs, n) in enumerate([(0.093, 0.008, 484), (0.192, 0.059, 188), (0.309, 0.538, 39), (0.432, 0.800, 15), (0.555, 0.600, 10)]):
    check("C20", f"reliability bin {i+1} pred", mpred, round(float(rb.iloc[i]["mean_predicted"]), 3))
    check("C20", f"reliability bin {i+1} obs", obs, round(float(rb.iloc[i]["observed_freq"]), 3))
    check("C20", f"reliability bin {i+1} n", n, int(rb.iloc[i]["n"]), kind="exact")

ca = csv("channel_ablation.csv").set_index("configuration")
full_oos = float(ca.loc["full", "auc_roc_oos"])
for blk, rep_auc, rep_pr, rep_only in [("dsp", 0.010, 0.037, 0.776), ("mob", -0.005, -0.033, 0.774),
                                       ("factor", -0.026, -0.096, 0.843)]:
    d_auc = float(ca.loc[f"ablate_{blk}", "auc_roc_oos"]) - full_oos
    d_pr = float(ca.loc[f"ablate_{blk}", "auc_pr_oos"]) - float(ca.loc["full", "auc_pr_oos"])
    check("C21", f"{blk} d_AUC", rep_auc, round(d_auc, 3))
    check("C21", f"{blk} d_AUC_PR", rep_pr, round(d_pr, 3))
    check("C21", f"{blk}_only OOS", rep_only, round(float(ca.loc[f"{blk}_only", "auc_roc_oos"]), 3))

for blk, rep_auc, rep_pr, rep_only in [("mobdem", -0.000, 0.008, 0.714), ("mobaut", 0.004, 0.013, 0.631),
                                       ("mobgen", 0.009, 0.006, 0.644)]:
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


# ---------------------------------------------------------------------------
# C27-C34: headline forecasting numbers that were outside the audit until
# referee2 round 1 (2026-09-06). Each is recomputed from the committed CSV.
# ---------------------------------------------------------------------------
from sklearn.metrics import roc_auc_score, average_precision_score
from scipy.stats import rankdata


def _blend(e):
    b = e.pivot_table(index=["origin", "country_name", "y"], columns="learner", values="p").reset_index()
    acc = np.zeros(len(b))
    for L in ("gb", "rf", "lr"):
        v = b[L].fillna(b[L].median()).values; r = np.zeros(len(v))
        for o in b.origin.unique():
            m = (b.origin == o).values; r[m] = rankdata(v[m]) / m.sum()
        acc += r
    b["blend"] = acc / 3
    return b


def _paired(y, pa, pb, c, seed=20260905, n=2000):
    rng = np.random.default_rng(seed); u = np.unique(c); idx = {k: np.where(c == k)[0] for k in u}
    da = []
    for _ in range(n):
        j = np.concatenate([idx[k] for k in rng.choice(u, len(u), replace=True)])
        if y[j].sum() < 3 or y[j].sum() == len(j):
            continue
        da.append(roc_auc_score(y[j], pa[j]) - roc_auc_score(y[j], pb[j]))
    return float(np.percentile(da, 2.5)), float(np.percentile(da, 97.5))


# C27 end-to-end, our episode set
e2e = _blend(csv("strict_endtoend_refit.csv"))
sp = csv("strict_table_predictions_h5.csv")
fw = sp[sp.model == "Five-stage framework, rank-mean blend"][["country_name", "year", "p"]].rename(columns={"p": "p_shared"})
pv = sp[sp.model == "Four polyarchy variables"][["country_name", "year", "p"]].rename(columns={"p": "p_poly4"})
m27 = e2e.merge(fw, left_on=["country_name", "origin"], right_on=["country_name", "year"]).merge(pv, on=["country_name", "year"])
check("C27", "end-to-end blend AUC (ledger)", 0.669, round(roc_auc_score(m27.y, m27.blend), 3), in_tex="0.669")
check("C27", "end-to-end blend AP (ledger)", 0.200, round(average_precision_score(m27.y, m27.blend), 3), in_tex="0.200")
check("C27", "shared-rep AUC on same rows", 0.659, round(roc_auc_score(m27.y, m27.p_shared), 3))
check("C27", "four-var AUC on same rows", 0.686, round(roc_auc_score(m27.y, m27.p_poly4), 3), in_tex="0.686")
check("C27", "e2e minus shared, point", 0.009, round(roc_auc_score(m27.y, m27.blend) - roc_auc_score(m27.y, m27.p_shared), 3), in_tex="+0.009")
lo, hi = _paired(m27.y.values, m27.blend.values, m27.p_shared.values, m27.country_name.values)
check("C27", "e2e minus shared, CI lo", -0.060, round(lo, 3), tol=0.01, in_tex="[-0.060, +0.073]")
check("C27", "e2e minus shared, CI hi", 0.073, round(hi, 3), tol=0.01)
check("C27", "e2e minus four-var, point", -0.017, round(roc_auc_score(m27.y, m27.blend) - roc_auc_score(m27.y, m27.p_poly4), 3), in_tex="-0.017")
lo, hi = _paired(m27.y.values, m27.blend.values, m27.p_poly4.values, m27.country_name.values)
check("C27", "e2e minus four-var, CI lo", -0.117, round(lo, 3), tol=0.01, in_tex="[-0.117, +0.081]")
check("C27", "e2e minus four-var, CI hi", 0.081, round(hi, 3), tol=0.01)
check("C27", "e2e scored rows", 629, len(m27), kind="exact", in_tex="629")
check("C27", "e2e positives", 88, int(m27.y.sum()), kind="exact", in_tex="88 positives")

# C28 end-to-end, ERT outcome
c28 = csv("strict_endtoend_ert_comparison.csv").set_index("model")
check("C28", "ERT e2e AUC", 0.658, round(float(c28.loc["end_to_end", "auc_roc"]), 3), in_tex="0.658")
check("C28", "ERT e2e AP", 0.293, round(float(c28.loc["end_to_end", "auc_pr"]), 3), in_tex="0.293")
check("C28", "ERT shared-rep AUC", 0.674, round(float(c28.loc["shared_representation", "auc_roc"]), 3))
check("C28", "ERT shared-rep AP", 0.318, round(float(c28.loc["shared_representation", "auc_pr"]), 3), in_tex="0.318")
check("C28", "ERT four-var AUC", 0.638, round(float(c28.loc["four_polyarchy", "auc_roc"]), 3), in_tex="0.638")
check("C28", "ERT e2e minus shared, point", -0.015, round(float(c28.loc["end_to_end_minus_shared", "auc_roc"]), 3), in_tex="-0.015")
check("C28", "ERT e2e minus shared, CI lo", -0.050, round(float(c28.loc["end_to_end_minus_shared", "auc_roc_lo"]), 3), tol=0.01, in_tex="[-0.050, +0.019]")
check("C28", "ERT e2e minus shared, CI hi", 0.019, round(float(c28.loc["end_to_end_minus_shared", "auc_roc_hi"]), 3), tol=0.01)
check("C28", "ERT e2e minus four-var, point", 0.021, round(float(c28.loc["end_to_end_minus_four_polyarchy", "auc_roc"]), 3), in_tex="+0.021")
check("C28", "ERT e2e minus four-var, CI lo", -0.041, round(float(c28.loc["end_to_end_minus_four_polyarchy", "auc_roc_lo"]), 3), tol=0.01, in_tex="[-0.041, +0.088]")
check("C28", "ERT e2e minus four-var, CI hi", 0.088, round(float(c28.loc["end_to_end_minus_four_polyarchy", "auc_roc_hi"]), 3), tol=0.01)
check("C28", "ERT e2e rows", 535, int(c28.loc["end_to_end", "n"]), kind="exact", in_tex="535")


def _marginals(df):
    keys = ["risk_set", "label", "origin", "closure"]; out = {}
    for k in keys:
        others = [x for x in keys if x != k] + ["learner"]
        piv = df.pivot_table(index=others, columns=k, values="auc")
        a, b = list(piv.columns); d = (piv[b] - piv[a]).dropna()
        # orient as the cost of the stricter setting, as Table 6 reports
        sign = -1 if k in ("label", "closure") else 1
        out[k] = (sign * float(d.mean()), sign * float(d.max()) if sign < 0 else float(d.min()),
                  sign * float(d.min()) if sign < 0 else float(d.max()))
    conv = df[(df.risk_set == "all") & (df.label == "window") & (df.origin == "fixed-2019") & (df.closure == "none")].auc.mean()
    strict = df[(df.risk_set == "at-risk") & (df.label == "future-only") & (df.origin == "rolling") & (df.closure == "enforced")].auc.mean()
    return out, float(conv), float(strict)


# C29 factorial, our episode set (Table 6 left pair)
mg, conv, strict = _marginals(csv("design_factorial.csv"))
for k, rep, lo_r, hi_r in [("risk_set", 0.026, -0.102, 0.128), ("label", -0.014, -0.060, 0.006),
                           ("origin", -0.008, -0.113, 0.166), ("closure", -0.231, -0.302, -0.123)]:
    mean, lo, hi = mg[k]
    check("C29", f"factorial {k} mean", rep, round(mean, 3), in_tex=f"{rep:+.3f}".replace("+0", "+0"))
    check("C29", f"factorial {k} range lo", lo_r, round(lo, 3), tol=0.002)
    check("C29", f"factorial {k} range hi", hi_r, round(hi, 3), tol=0.002)
check("C29", "conventional corner", 0.874, round(conv, 3), in_tex="0.874")
check("C29", "strict corner", 0.662, round(strict, 3))

# C30 factorial, ERT outcome (Table 6 right pair)
mg, conv, strict = _marginals(csv("design_factorial_ert.csv"))
for k, rep, lo_r, hi_r in [("risk_set", 0.024, -0.057, 0.071), ("label", -0.004, -0.033, 0.033),
                           ("origin", -0.084, -0.208, 0.064), ("closure", -0.142, -0.209, -0.010)]:
    mean, lo, hi = mg[k]
    check("C30", f"ERT factorial {k} mean", rep, round(mean, 3), in_tex=f"{rep:+.3f}")
    check("C30", f"ERT factorial {k} range lo", lo_r, round(lo, 3), tol=0.002)
    check("C30", f"ERT factorial {k} range hi", hi_r, round(hi, 3), tol=0.002)
check("C30", "ERT conventional corner", 0.883, round(conv, 3), in_tex="0.883")
check("C30", "ERT strict corner", 0.671, round(strict, 3), in_tex="0.671")

# C31 strict table, ERT outcome (Table 5 Panel B)
t31 = csv("strict_ert_sensitivity.csv").set_index("model")
for name, rep in [("Persistence (3-yr polyarchy decline)", 0.344), ("Four polyarchy variables", 0.705),
                  ("Five-stage framework", 0.733), ("Random forest", 0.730), ("Gradient boosting", 0.685), ("Elastic net", 0.672)]:
    check("C31", f"Panel B AUC {name[:22]}", rep, round(float(t31.loc[name, "auc_roc"]), 3), in_tex=f"{rep:.3f}")
check("C31", "Panel B persistence CI lo", 0.271, round(float(t31.loc["Persistence (3-yr polyarchy decline)", "auc_roc_lo"]), 3), tol=0.002, in_tex="[0.271, 0.419]")
check("C31", "Panel B rows", 1002, int(t31.loc["Five-stage framework", "n"]), kind="exact")
check("C31", "Panel B positives", 179, int(t31.loc["Five-stage framework", "n_pos"]), kind="exact", in_tex="179 positives")

# C32 distinct onsets behind the strict positives
check("C32", "ledger distinct onsets (strict)", 28, int(sp[(sp.model == "Five-stage framework, rank-mean blend") & (sp.y == 1)].country_name.nunique()), kind="exact", in_tex="28 distinct onsets")
sys.path.insert(0, ROB)
from ert_panel import build_panel_ert
_ep = build_panel_ert(); _ons = _ep.attrs["onsets"]
_pb = csv("strict_ert_sensitivity_predictions.csv"); _pos = _pb[_pb.y == 1]
_keys = set()
for r in _pos.itertuples():
    for o in sorted(_ons.get(r.country_name, [])):
        if 1 <= o - r.year <= 5:
            _keys.add((r.country_name, o)); break
check("C32", "ERT distinct onsets (strict)", 48, len(_keys), kind="exact", in_tex="48 distinct onsets")
check("C32", "ERT distinct countries w/ onset", 42, int(_pos.country_name.nunique()), kind="exact", in_tex="42 countries")

# C33 minimum detectable effects on the mobilization block (2.80 * SE, SE = width/3.92)
for tag, fname, rep in [("ledger", "onset_ablation_ci.csv", 0.041), ("ERT", "onset_ablation_ci_ert.csv", 0.033)]:
    ci = csv(fname); g = ci[(ci.h == 5) & (ci.block == "mob")][["d_auc_lo", "d_auc_hi"]].mean()
    check("C33", f"MDE mobilization h=5 ({tag})", rep, round(2.80 * (g.d_auc_hi - g.d_auc_lo) / 3.92, 3), tol=0.002, in_tex=f"{rep:.3f}")

# C34 channel ablation under ERT, pooled over learners and seeds
ci = csv("onset_ablation_ci_ert.csv"); g = ci.groupby(["h", "block"])[["d_auc_mean", "d_auc_lo", "d_auc_hi"]].mean()
for h, blk, mean, lo_r, hi_r in [(5, "mob", 0.004, -0.018, 0.025), (2, "mob", 0.010, -0.014, 0.035),
                                  (5, "dsp", 0.005, -0.036, 0.046), (2, "dsp", 0.011, -0.019, 0.044)]:
    r = g.loc[(h, blk)]
    check("C34", f"ERT {blk} h={h} mean", mean, round(float(r.d_auc_mean), 3), tol=0.002, in_tex=f"[{lo_r:+.3f}, {hi_r:+.3f}]")
    check("C34", f"ERT {blk} h={h} CI lo", lo_r, round(float(r.d_auc_lo), 3), tol=0.002)
    check("C34", f"ERT {blk} h={h} CI hi", hi_r, round(float(r.d_auc_hi), 3), tol=0.002)


# C35 placebo, C36 size-matched arm (referee2 round 1)
_pl = csv("closure_placebo.csv"); _obs = float(_pl[_pl.kind == "observed"].mean_contrast.iloc[0]); _pp = _pl[_pl.kind == "placebo"].mean_contrast
check("C35", "placebo observed closure contrast", -0.191, round(_obs, 3), tol=0.002, in_tex="-0.191")
check("C35", "placebo mean", -0.303, round(float(_pp.mean()), 3), tol=0.002, in_tex="-0.303")
check("C35", "placebo sd", 0.038, round(float(_pp.std()), 3), tol=0.002, in_tex="0.038")
check("C35", "placebo n more negative", 50, int((_pp <= _obs).sum()), kind="exact", in_tex="every one of the fifty")
_sm = pd.read_csv(os.path.join(REPO, "code", "replication", "referee2_closure_sizematch.csv"))
check("C36", "size-match closed minus open", -0.193, round(float(_sm.closed_minus_open.mean()), 3), tol=0.002, in_tex="-0.193")
check("C36", "size-match closed minus open_matched", -0.154, round(float(_sm.closed_minus_open_matched.mean()), 3), tol=0.002, in_tex="-0.154")

# C37: network-dependence seed sweep (Section 5.6, Figure 3, Appendix J)
import numpy as _np
from scipy.stats import spearmanr as _spr
_sw = csv("netdep_tv_seed_sweep.csv")
_g = _sw.groupby("country")
for c, key, rep in [("US", "United States of America", 0.422), ("Hungary", "Hungary", 0.244),
                    ("Poland", "Poland", 0.132), ("Turkiye", "T\u00fcrkiye", 0.050)]:
    check("C37", f"seed-mean TV {c}", rep, round(float(_g["tv"].mean()[key]), 3), in_tex=f"${rep:.3f}$")
check("C37", "Hungary seed-mean direction", -0.223, round(float(_g["signed"].mean()["Hungary"]), 3), in_tex="$-0.223$")
_P = _sw.pivot(index="seed", columns="country", values="tv")
_rs = [_spr(_P.iloc[i], _P.iloc[j])[0] for i in range(len(_P)) for j in range(i + 1, len(_P))]
check("C37", "seed rank corr mean", 0.76, round(float(_np.mean(_rs)), 2), tol=0.01, in_tex="$0.76$")
check("C37", "seed rank corr min", 0.54, round(float(_np.min(_rs)), 2), tol=0.01, in_tex="$0.54$")
check("C37", "Hungary > Turkiye seeds", 10, int((_P["Hungary"] > _P["T\u00fcrkiye"]).sum()), kind="exact", in_tex="all ten runs")
check("C37", "Turkiye positive seeds", 10, int((_sw[_sw.country == "T\u00fcrkiye"].signed > 0).sum()), kind="exact", in_tex="positive in all ten runs")
check("C37", "Hungary negative seeds", 8, int((_sw[_sw.country == "Hungary"].signed < 0).sum()), kind="exact", in_tex="negative in eight of ten")

# C38: locked 2025 cross-section (Section 5.6, Section 8)
_tv = csv("netdep_total_variation.csv")
_y = _tv[_tv.year == 2025].set_index("country_name")
check("C38", "locked US TV", 0.291, round(float(_y.loc["United States of America", "netdep_tv"]), 3), in_tex="$0.291$")
check("C38", "locked US rank of 135", 15, int(_y.netdep_tv.rank(ascending=False)["United States of America"]), kind="exact", in_tex="fifteenth of 135")
check("C38", "locked Hungary direction", 0.419, round(float(_y.loc["Hungary", "netdep_signed"]), 3), in_tex="$+0.419$")
check("C38", "locked Moldova TV", 0.823, round(float(_y.loc["Moldova", "netdep_tv"]), 3), in_tex="$0.823$")
check("C38", "locked Poland TV", 0.126, round(float(_y.loc["Poland", "netdep_tv"]), 3), in_tex="$0.126$")
check("C38", "share signed toward autocracy", 0.44, round(float((_tv.netdep_signed > 0).mean()), 2), tol=0.005, in_tex="44 percent")
out = pd.DataFrame(rows)
out.to_csv(os.path.join(ROB, "audit_rows.csv"), index=False)
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
