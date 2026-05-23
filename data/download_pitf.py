"""
F5: PITF / IMF-style economic and demographic features from World Bank WDI.

Indicators (Goldstone 2010 PITF, IMF 2024 fragility):
  - Infant mortality (SP.DYN.IMRT.IN)
  - Inflation, CPI YoY (FP.CPI.TOTL.ZG)
  - Food production index (AG.PRD.FOOD.XD)
  - External debt to GNI (DT.DOD.DECT.GN.ZS)
  - Working-age share (proxy for youth bulge)

Writes data/macro_pitf.csv merging on (iso3, year).
"""

import os
import pandas as pd
import wbgapi as wb

OUT = os.path.join(os.path.dirname(__file__), "macro_pitf.csv")

INDICATORS = {
    "SP.DYN.IMRT.IN":   "infant_mortality",
    "FP.CPI.TOTL.ZG":   "inflation_yoy",
    "AG.PRD.FOOD.XD":   "food_prod_index",
    "DT.DOD.DECT.GN.ZS":"ext_debt_gni",
    "SP.POP.1564.TO.ZS":"work_age_share",
}


def fetch_one(code, name, years):
    df = wb.data.DataFrame(code, time=years, labels=False)
    df = df.reset_index()
    first = df.columns[0]
    df = df.rename(columns={first: "iso3"})
    year_cols = [c for c in df.columns if c != "iso3"]
    df = df.melt(id_vars=["iso3"], value_vars=year_cols, var_name="yr", value_name=name)

    def _to_year(v):
        s = str(v)
        return int(s.replace("YR", "")) if s.replace("YR", "").isdigit() else None

    df["year"] = df["yr"].apply(_to_year)
    df = df.dropna(subset=["year"])
    df["year"] = df["year"].astype(int)
    return df[["iso3", "year", name]]


def main():
    years = range(1960, 2026)
    print(f"Fetching {len(INDICATORS)} WDI indicators 1960-2025...")
    parts = []
    for code, name in INDICATORS.items():
        try:
            print(f"  {code} ({name})...")
            parts.append(fetch_one(code, name, years))
        except Exception as e:
            print(f"    failed: {e}")

    if not parts:
        raise RuntimeError("No WDI indicators fetched")

    df = parts[0]
    for p in parts[1:]:
        df = df.merge(p, on=["iso3", "year"], how="outer")

    if "work_age_share" in df.columns:
        df["youth_bulge_proxy"] = df["work_age_share"] * 0.4

    df = df.sort_values(["iso3", "year"]).reset_index(drop=True)
    for col in df.columns:
        if col in {"iso3", "year"}:
            continue
        df[col] = df.groupby("iso3")[col].ffill()

    df.to_csv(OUT, index=False)
    print(f"\nWrote {OUT}: {len(df)} rows, {df['iso3'].nunique()} countries, "
          f"{df['year'].min()}-{df['year'].max()}")
    print(f"Indicators: {[c for c in df.columns if c not in {'iso3','year'}]}")


if __name__ == "__main__":
    main()
