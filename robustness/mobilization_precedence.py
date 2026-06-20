"""Temporal precedence of the mobilization signal (answer to 'prediction is not
mechanism'). We make a predictive, not causal, claim, but a leading indicator
must at least PRECEDE the event rather than coincide with or follow it. The
reverse-causality worry is that mobilization is a RESPONSE to visible erosion, not
an antecedent. Three checks, all out of the model:

  (A) Event study: mean standardized mobilization vs digital-control signal in the
      five years BEFORE autocratization onset (event time tau = -5..0).
  (B) Pre-onset elevation: is each channel already elevated at tau in [-3,-1]
      relative to non-onset baseline country-years?
  (C) Lead comparison: in how many episodes does mobilization cross an alert level
      EARLIER than digital control, and what is the median lead?
  (D) Lead-lag direction: within-country cross-correlation between mobilization and
      the year-over-year change in the democratic factor. If mobilization leads
      decline, mob_t predicts factor decline at t+k more strongly than past decline
      predicts mobilization at t+k.
"""

import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "4")

import warnings

warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
from scipy import stats

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUTPUT_DIR = os.path.dirname(os.path.abspath(__file__))
MOB = ["v2cagenmob", "v2caconmob", "v2cademmob", "v2caautmob"]
DIG = ["v2smgovdom", "v2smpardom", "v2smfordom", "v2smgovfilprc", "v2smgovsmmon"]
THRESH = 0.5


def zcomp(e, cols):
    Z = (e[cols] - e[cols].mean()) / e[cols].std(ddof=0)
    return Z.mean(axis=1)


def main():
    print("=" * 70)
    print("MOBILIZATION TEMPORAL PRECEDENCE")
    print("=" * 70)
    e = pd.read_csv(os.path.join(REPO, "stage5_ews", "ews_signals.csv"))
    e = e.sort_values(["country_text_id", "year"]).reset_index(drop=True)
    e["mob_z"] = zcomp(e, MOB)
    e["dig_z"] = zcomp(e, DIG)
    look = {(r.country_text_id, int(r.year)): (r.mob_z, r.dig_z) for r in e.itertuples()}

    onsets = []
    for c, g in e.groupby("country_text_id"):
        po = g["is_postonset"].astype(bool).values
        yrs = g["year"].astype(int).values
        for k in range(len(po)):
            if po[k] and (k == 0 or not po[k - 1]):
                onsets.append((c, yrs[k]))
    print(f"  {len(onsets)} autocratization onsets identified\n")

    print("  (A) Event study, mean standardized signal by event time:")
    print("      tau   mobilization   digital control   n")
    out = {}
    for tau in range(-5, 1):
        mv = [look[(c, t0 + tau)][0] for c, t0 in onsets if (c, t0 + tau) in look]
        dv = [look[(c, t0 + tau)][1] for c, t0 in onsets if (c, t0 + tau) in look]
        print(f"      {tau:+d}    {np.mean(mv):+.3f}          {np.mean(dv):+.3f}         {len(mv)}")
        out[f"mob_tau{tau}"], out[f"dig_tau{tau}"] = np.mean(mv), np.mean(dv)

    pre_mob = [look[(c, t0 + tau)][0] for c, t0 in onsets for tau in (-3, -2, -1) if (c, t0 + tau) in look]
    pre_dig = [look[(c, t0 + tau)][1] for c, t0 in onsets for tau in (-3, -2, -1) if (c, t0 + tau) in look]
    base = e[(~e["is_postonset"].astype(bool)) & (e["label"] == 0)]
    tm, pm = stats.ttest_ind(pre_mob, base["mob_z"].dropna(), equal_var=False)
    td, pd_ = stats.ttest_ind(pre_dig, base["dig_z"].dropna(), equal_var=False)
    print(f"\n  (B) Pre-onset [-3,-1] elevation vs baseline:")
    print(f"      mobilization    mean {np.mean(pre_mob):+.3f} vs base {base['mob_z'].mean():+.3f}  t={tm:.2f} p={pm:.4f}")
    print(f"      digital control mean {np.mean(pre_dig):+.3f} vs base {base['dig_z'].mean():+.3f}  t={td:.2f} p={pd_:.4f}")
    out.update(mob_pre=np.mean(pre_mob), mob_pre_p=pm, dig_pre=np.mean(pre_dig), dig_pre_p=pd_)

    lead_m, lead_d = [], []
    for c, t0 in onsets:
        lm = next((-tau for tau in range(-5, 0) if (c, t0 + tau) in look and look[(c, t0 + tau)][0] > THRESH), np.nan)
        ld = next((-tau for tau in range(-5, 0) if (c, t0 + tau) in look and look[(c, t0 + tau)][1] > THRESH), np.nan)
        lead_m.append(lm)
        lead_d.append(ld)
    lead_m, lead_d = np.array(lead_m), np.array(lead_d)
    both = ~np.isnan(lead_m) & ~np.isnan(lead_d)
    mob_earlier = int(np.sum(lead_m[both] >= lead_d[both]))
    print(f"\n  (C) Lead time (years before onset crossing +{THRESH} SD):")
    print(f"      mobilization fires in {np.sum(~np.isnan(lead_m))}/{len(onsets)} episodes, median lead {np.nanmedian(lead_m):.1f} yr")
    print(f"      digital control fires in {np.sum(~np.isnan(lead_d))}/{len(onsets)} episodes, median lead {np.nanmedian(lead_d):.1f} yr")
    print(f"      mobilization crosses at least as early as digital control in {mob_earlier}/{int(both.sum())} episodes with both")
    out.update(mob_fires=int(np.sum(~np.isnan(lead_m))), dig_fires=int(np.sum(~np.isnan(lead_d))),
               mob_med_lead=float(np.nanmedian(lead_m)), dig_med_lead=float(np.nanmedian(lead_d)))

    fwd, rev = [], []
    for c, g in e.groupby("country_text_id"):
        g = g.sort_values("year")
        m = (g["mob_z"] - g["mob_z"].mean()).values
        d = (g["f1_change"] - g["f1_change"].mean()).values
        if len(g) < 8 or np.nanstd(m) == 0 or np.nanstd(d) == 0:
            continue
        for k in (1, 2, 3):
            if len(g) > k:
                a, b = m[:-k], d[k:]
                ok = ~np.isnan(a) & ~np.isnan(b)
                if ok.sum() > 3:
                    fwd.append(np.corrcoef(a[ok], b[ok])[0, 1])
                a2, b2 = d[:-k], m[k:]
                ok2 = ~np.isnan(a2) & ~np.isnan(b2)
                if ok2.sum() > 3:
                    rev.append(np.corrcoef(a2[ok2], b2[ok2])[0, 1])
    print(f"\n  (D) Lead-lag direction (within-country, corr):")
    print(f"      mobilization_t -> factor change_t+k : mean r = {np.mean(fwd):+.3f}  (negative = mobilization precedes decline)")
    print(f"      factor change_t -> mobilization_t+k : mean r = {np.mean(rev):+.3f}")
    print(f"      => mobilization {'LEADS' if abs(np.mean(fwd)) > abs(np.mean(rev)) else 'does not clearly lead'} the decline")
    out.update(fwd_r=float(np.mean(fwd)), rev_r=float(np.mean(rev)))

    pd.DataFrame([out]).to_csv(os.path.join(OUTPUT_DIR, "mobilization_precedence_results.csv"), index=False)
    print(f"\nSaved to robustness/mobilization_precedence_results.csv")


if __name__ == "__main__":
    main()
