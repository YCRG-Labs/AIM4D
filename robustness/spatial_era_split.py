"""C.7 specification ladder split at 2005.

Section C.5 reports that the contiguity channel is carried by the pre-2005
democratization wave and falls to near zero across the backsliding era. Section
C.7 reports a full-panel autoregressive parameter of 0.333 under the Lee best
instrument. Those are estimated on different samples, so this script re-runs the
C.7 ladder separately on the pre-2005 and post-2005 halves and reports the
autoregressive parameter in each. That is the comparison needed to say whether
the two sections agree or conflict.

Outputs robustness/spatial_era_split.csv.
"""

import os
import sys
import warnings

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from contagion_galton import build_inputs, build_W_contiguity
from spatial_models import YearBlockW, fit_all, ols, EXOG, MACRO, N_BOOT

OUT = os.path.dirname(os.path.abspath(__file__))


def panel():
    df, countries, years, Y = build_inputs()
    W, _ = build_W_contiguity(countries)
    mac = pd.read_csv(MACRO).rename(columns={"iso3": "country_text_id"})
    mac = mac.sort_values(["country_text_id", "year"])
    for c in EXOG:
        mac[c] = mac.groupby("country_text_id")[c].shift(1)
    rows = []
    for ti in range(1, len(years)):
        t = years[ti]
        dy = Y[:, ti] - Y[:, ti - 1]
        for i, c in enumerate(countries):
            if np.isnan(dy[i]) or np.isnan(Y[i, ti - 1]):
                continue
            rows.append((c, t, dy[i], Y[i, ti - 1]))
    p = pd.DataFrame(rows, columns=["country_text_id", "year", "dy", "y_lag"])
    p = p.merge(mac[["country_text_id", "year"] + EXOG], on=["country_text_id", "year"], how="left")
    return p, W, countries


def run(p, W, countries, tag):
    p = p.copy()
    p[EXOG] = p[EXOG].apply(lambda s: (s - s.mean()) / s.std())
    p = p.dropna(subset=["dy", "y_lag"] + EXOG).reset_index(drop=True)
    cid, yr = p["country_text_id"].values, p["year"].values
    Wop = YearBlockW(cid, yr, W, countries)
    Xcols = ["y_lag"] + EXOG
    yrd = pd.get_dummies(p["year"], drop_first=True).values.astype(float)
    Xe = np.column_stack([np.ones(len(p))] + [p[c].values for c in Xcols] + [yrd])
    WX = np.column_stack([Wop(p[c].values) for c in Xcols])
    y = p["dy"].values
    print(f"\n[{tag}] n = {len(p)} country-years, {p.country_text_id.nunique()} countries, "
          f"{yr.min()}--{yr.max()}")
    base = fit_all(y, Xe, WX, Wop, cid)

    # Same residual bootstrap as spatial_models.main(): hold the network fixed,
    # resample only the innovations, and generate from the estimator's own fit.
    rng = np.random.default_rng(20260905)

    def neumann(vec, coef, iters=40):
        acc, term = vec.copy(), vec.copy()
        for _ in range(iters):
            term = coef * Wop(term)
            acc = acc + term
            if np.max(np.abs(term)) < 1e-10:
                break
        return acc

    Wy_obs = Wop(y)
    draws = []
    rho_a = float(np.clip(base["SAR_lee"]["rho"], -0.9, 0.9))
    beta_a, u_a = ols(y - rho_a * Wy_obs, Xe)
    u_a = u_a - u_a.mean()
    for _ in range(N_BOOT):
        e = u_a * rng.choice([-1.0, 1.0], size=len(u_a))
        y_star = neumann(Xe @ beta_a + e, rho_a)
        try:
            r = fit_all(y_star, Xe, WX, Wop, cid)
        except Exception:
            continue
        v = r["SAR_lee"]["rho"]
        if np.isfinite(v):
            draws.append(v)
    lo, hi = (float(np.percentile(draws, 2.5)), float(np.percentile(draws, 97.5))) if draws else (float("nan"),) * 2
    base["SAR_lee_ci"] = {"lo": round(lo, 4), "hi": round(hi, 4), "n_boot": len(draws)}
    return base, len(p), int(yr.min()), int(yr.max())


def main():
    p, W, countries = panel()
    specs = [("full panel", p),
             ("pre-2005", p[p.year <= 2005]),
             ("post-2005", p[p.year > 2005])]
    rows = []
    for tag, sub in specs:
        out, n, y0, y1 = run(sub, W, countries, tag)
        for k, v in out.items():
            rows.append({"sample": tag, "n": n, "y0": y0, "y1": y1,
                         "param": k, "value": v})
        print(f"[{tag}] " + "  ".join(
            f"{k}={v}" for k, v in out.items()
            if isinstance(v, (int, float)) and ("rho" in k or "F" in k)))
    d = pd.DataFrame(rows)
    d.to_csv(os.path.join(OUT, "spatial_era_split.csv"), index=False)
    print("\nWrote spatial_era_split.csv")


if __name__ == "__main__":
    main()
