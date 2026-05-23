"""
G7: catch22 recurrent time-series features per country.

catch22 (Lubba et al. 2019) is a 22-feature subset of the hctsa library
that captures rolling-window slope, peak count, time-reversal asymmetry,
and other temporal-structure indicators. It is the field-standard
fast-and-interpretable summary of a time series.

For each country we compute catch22 over a rolling 8-year window of the
v2x_polyarchy and v2x_libdem time series and export the features per
country-year.

Output: data/catch22_features.csv with one column per (variable, statistic).
"""

import os
import numpy as np
import pandas as pd

try:
    from pycatch22 import catch22_all
except ImportError:
    raise SystemExit("Install pycatch22: pip install pycatch22")

DATA = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(DATA, "catch22_features.csv")
WINDOW = 10


def rolling_catch22(series, window=WINDOW):
    """Return list of 22-feature dicts (one per t with at least window obs)."""
    n = len(series)
    out = []
    for t in range(n):
        if t + 1 < window:
            out.append({f"c22_{i}": np.nan for i in range(22)})
            continue
        chunk = series[max(0, t - window + 1):t + 1]
        if np.all(np.isnan(chunk)) or np.nanstd(chunk) < 1e-10:
            out.append({f"c22_{i}": np.nan for i in range(22)})
            continue
        try:
            res = catch22_all(chunk.tolist())
            out.append({f"c22_{i}": res["values"][i] for i in range(22)})
        except Exception:
            out.append({f"c22_{i}": np.nan for i in range(22)})
    return out


def main():
    print("Loading V-Dem polyarchy + libdem series...")
    df = pd.read_csv(
        os.path.join(DATA, "vdem_v16.csv"), low_memory=False,
        usecols=["country_text_id", "year", "v2x_polyarchy", "v2x_libdem"],
    )
    df = df.dropna(subset=["v2x_polyarchy"]).sort_values(["country_text_id", "year"])

    rows = []
    n_countries = df["country_text_id"].nunique()
    for i, (cid, grp) in enumerate(df.groupby("country_text_id"), 1):
        grp = grp.sort_values("year").reset_index(drop=True)
        years = grp["year"].values
        if len(grp) < WINDOW:
            continue

        poly = grp["v2x_polyarchy"].values
        libdem = grp["v2x_libdem"].ffill().fillna(0.0).values

        poly_feats = rolling_catch22(poly)
        libdem_feats = rolling_catch22(libdem)

        for t, y in enumerate(years):
            row = {"country_text_id": cid, "year": int(y)}
            for k, v in poly_feats[t].items():
                row[f"poly_{k}"] = v
            for k, v in libdem_feats[t].items():
                row[f"libdem_{k}"] = v
            rows.append(row)
        if i % 20 == 0:
            print(f"  {i}/{n_countries} countries done")

    out = pd.DataFrame(rows)
    out = out.dropna(axis=1, how="all")
    out.to_csv(OUT, index=False)
    print(f"\nWrote {OUT}: {len(out)} rows, {out.shape[1]-2} features, "
          f"{out['country_text_id'].nunique()} countries")


if __name__ == "__main__":
    main()
