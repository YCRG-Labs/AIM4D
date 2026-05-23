"""
G9: Change-point features via PELT (Pruned Exact Linear Time) — PROSPECTIVE.

For each country-year (c, T), we run PELT on the TRAILING 30-year window
[T-29, T] only. That makes every feature value at year T derived strictly
from data available by year T — no future leakage. (The previous version ran
PELT once globally over 1789-2025, so years_since_break at year T could
reflect breakpoint positions PELT chose using post-T data.)

Output: data/changepoints.csv with columns
  country_text_id, year, years_since_break_poly, years_since_break_libdem,
  break_in_last_3yr_poly, break_in_last_3yr_libdem
"""

import os
import numpy as np
import pandas as pd

try:
    import ruptures as rpt
except ImportError:
    raise SystemExit("Install ruptures: pip install ruptures")

DATA = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(DATA, "changepoints.csv")
WINDOW = 30


def detect_breaks_in_window(series, penalty=0.05, model="l2", min_size=4):
    """Return sorted list of break-year indices (within-window) for a 1-D series."""
    x = np.asarray(series, dtype=float)
    if len(x) < min_size * 2:
        return []
    try:
        algo = rpt.Pelt(model=model, min_size=min_size, jump=1).fit(x)
        bkps = algo.predict(pen=penalty)
        return [b for b in bkps if 0 < b < len(x)]
    except Exception:
        return []


def years_since_last_break_prospective(series, years, window=WINDOW,
                                        penalty=0.05, min_size=4):
    """
    For each index t, run PELT on series[max(0, t-window+1) : t+1].
    Return (years_since_last_break, break_in_last_3_years) arrays of length n.
    """
    n = len(series)
    yss = np.full(n, 99, dtype=int)
    win3 = np.zeros(n, dtype=int)
    for t in range(n):
        start = max(0, t - window + 1)
        chunk = series[start:t + 1]
        chunk_years = years[start:t + 1]
        valid = ~np.isnan(chunk)
        if valid.sum() < min_size * 2:
            continue
        bkps = detect_breaks_in_window(chunk[valid], penalty=penalty, min_size=min_size)
        valid_years = chunk_years[valid]
        break_years = [int(valid_years[b]) for b in bkps if b < len(valid_years)]
        if break_years:
            last_break = max(break_years)
            yss[t] = int(years[t]) - last_break
            if 0 <= int(years[t]) - last_break <= 2:
                win3[t] = 1
    return yss, win3


def main():
    print("Loading V-Dem polyarchy and libdem time series...")
    df = pd.read_csv(
        os.path.join(DATA, "vdem_v16.csv"), low_memory=False,
        usecols=["country_text_id", "year", "v2x_polyarchy", "v2x_libdem"],
    )
    df = df.dropna(subset=["v2x_polyarchy"]).sort_values(["country_text_id", "year"])

    rows = []
    countries = df["country_text_id"].unique()
    n_total = len(countries)
    for i, cid in enumerate(countries, 1):
        grp = df[df["country_text_id"] == cid].sort_values("year").reset_index(drop=True)
        years = grp["year"].values.astype(int)
        if len(grp) < 8:
            continue

        poly = grp["v2x_polyarchy"].values
        libdem = grp["v2x_libdem"].ffill().fillna(0.0).values

        yss_p, w3_p = years_since_last_break_prospective(poly, years)
        yss_l, w3_l = years_since_last_break_prospective(libdem, years)

        for t in range(len(years)):
            rows.append({
                "country_text_id": cid,
                "year": int(years[t]),
                "years_since_break_poly": int(yss_p[t]),
                "years_since_break_libdem": int(yss_l[t]),
                "break_in_last_3yr_poly": int(w3_p[t]),
                "break_in_last_3yr_libdem": int(w3_l[t]),
            })

        if i % 20 == 0:
            print(f"  {i}/{n_total} countries done")

    out = pd.DataFrame(rows)
    out.to_csv(OUT, index=False)
    print(f"Wrote {OUT}: {len(out)} rows, {out['country_text_id'].nunique()} countries, "
          f"{out['year'].min()}-{out['year'].max()} (prospective {WINDOW}-yr window)")


if __name__ == "__main__":
    main()
