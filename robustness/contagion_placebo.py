"""Additional identification checks for the spatial-diffusion / contagion channel,
on top of causal_real_data.py (which already runs the level-FE artifact, the
change-spec +/- year FE contrast, 2SLS, and a node-permutation placebo).

  temporal_placebo : the spatial-lag coefficient should be ~null in the pre-wave
                     period and significant in the backsliding era (Egami 2024
                     structural-stationarity logic).
  slx_predetermined: dY on neighbors' PREDETERMINED lagged level (W*y_{t-1}),
                     which is IV-free and avoids the simultaneity of W*dY
                     (Halleck-Vega & Elhorst SLX).
  common_shock     : alpha on W*dY survives controlling for the global mean change
                     (a common-shock proxy weaker than full year FE).
  first_stage_F    : the weak-IV first-stage F for the 2SLS instrument W*y_{t-1}
                     (transparency on why we frame as predictive, not causal).

Reuses build_W and the factor/state inputs from causal_real_data.
"""

import os
import sys
import warnings

warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from causal_real_data import build_W, FACT, STATE

OUTPUT_DIR = os.path.dirname(os.path.abspath(__file__))
SPLIT_YEAR = 2005


def build_spatial_panel():
    fac = pd.read_csv(FACT)[["country_text_id", "year", "factor_1"]]
    stt = pd.read_csv(STATE)[["country_text_id", "year", "state"]]
    df = fac.merge(stt, on=["country_text_id", "year"], how="inner").dropna()
    countries = sorted(df["country_text_id"].unique())
    W = build_W(countries)
    years = sorted(df["year"].unique())
    Y = df.pivot(index="country_text_id", columns="year", values="factor_1").reindex(countries).values

    rows = []
    for ti in range(1, len(years)):
        t = years[ti]
        yprev, ycur = Y[:, ti - 1], Y[:, ti]
        dy_vec = ycur - yprev
        wdy = W @ np.where(np.isnan(dy_vec), 0.0, dy_vec)
        wy_lag = W @ np.where(np.isnan(yprev), 0.0, yprev)
        for i, c in enumerate(countries):
            if np.isnan(ycur[i]) or np.isnan(yprev[i]):
                continue
            rows.append((c, t, ycur[i] - yprev[i], yprev[i], wdy[i], wy_lag[i]))
    p = pd.DataFrame(rows, columns=["country_text_id", "year", "dy", "y_lag", "wdy", "wy_lag"])
    p = p.merge(df[["country_text_id", "year", "state"]], on=["country_text_id", "year"])
    p["global_dy"] = p.groupby("year")["dy"].transform("mean")
    return p


def clustered(formula, data):
    return smf.ols(formula, data=data).fit(cov_type="cluster",
                                           cov_kwds={"groups": data["country_text_id"]})


def main():
    print("=" * 70)
    print("CONTAGION: additional identification checks")
    print("=" * 70)
    p = build_spatial_panel()
    print(f"  panel: {len(p)} country-years, {p['country_text_id'].nunique()} countries\n")
    out = {}

    print("-" * 70)
    print(f"[A] Temporal placebo: change spec by period (split at {SPLIT_YEAR})")
    print("-" * 70)
    for label, sub in [("pre  (< %d)" % SPLIT_YEAR, p[p["year"] < SPLIT_YEAR]),
                       ("post (>=%d)" % SPLIT_YEAR, p[p["year"] >= SPLIT_YEAR])]:
        if sub["state"].nunique() < 2 or len(sub) < 50:
            print(f"    {label}: insufficient sample"); continue
        m = clustered("dy ~ y_lag + wdy + C(state)", sub)
        print(f"    {label}: alpha={m.params['wdy']:+.4f}  t={m.tvalues['wdy']:.2f}  p={m.pvalues['wdy']:.3f}")
        out[f"placebo_{'pre' if '<' in label else 'post'}_alpha"] = m.params["wdy"]
        out[f"placebo_{'pre' if '<' in label else 'post'}_p"] = m.pvalues["wdy"]

    print("\n" + "-" * 70)
    print("[B] SLX / predetermined spatial lag: dY ~ y_lag + W*y_{t-1} (IV-free)")
    print("-" * 70)
    m_slx = clustered("dy ~ y_lag + wy_lag + C(state)", p)
    print(f"    alpha(W*y_lag)={m_slx.params['wy_lag']:+.4f}  t={m_slx.tvalues['wy_lag']:.2f}  p={m_slx.pvalues['wy_lag']:.2e}")
    out["slx_alpha"] = m_slx.params["wy_lag"]; out["slx_p"] = m_slx.pvalues["wy_lag"]

    print("\n" + "-" * 70)
    print("[C] Common-shock control: dY ~ y_lag + W*dY + global_dY + C(state)")
    print("-" * 70)
    m_cs = clustered("dy ~ y_lag + wdy + global_dy + C(state)", p)
    print(f"    alpha(W*dY)={m_cs.params['wdy']:+.4f}  t={m_cs.tvalues['wdy']:.2f}  p={m_cs.pvalues['wdy']:.2e}  "
          f"(survives common-shock proxy)")
    out["commonshock_alpha"] = m_cs.params["wdy"]; out["commonshock_p"] = m_cs.pvalues["wdy"]

    print("\n" + "-" * 70)
    print("[D] Weak-IV first-stage F for the 2SLS instrument W*y_{t-1}")
    print("-" * 70)
    m_fs = clustered("wdy ~ y_lag + wy_lag + C(state)", p)
    F = float(m_fs.f_test("wy_lag = 0").fvalue)
    print(f"    first-stage F on W*y_lag = {F:.2f}  "
          f"({'WEAK (<10): clean causal ID out of reach' if F < 10 else 'adequate'})")
    out["first_stage_F"] = F

    pd.DataFrame([out]).to_csv(os.path.join(OUTPUT_DIR, "contagion_placebo_results.csv"), index=False)
    print(f"\nSaved to robustness/contagion_placebo_results.csv")


if __name__ == "__main__":
    main()
