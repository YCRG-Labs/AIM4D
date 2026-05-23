"""
Download Freedom House (Freedom in the World) + Polity5 for the measurement-
invariance check (robustness/measurement_invariance.py).

FH "All Data" has 25 sub-question scores (A1..G4) but only from 2013 onward.
Polity5 has 6 component variables (xrcomp, xropen, xconst, parreg, parcomp)
1800-2018.

FH is CC-BY; Polity5 is public (Center for Systemic Peace).

Outputs:
  data/fh_subscores.csv   (country, year, A1..G4)
  data/polity5.csv        (ccode, country, year, xrcomp, xropen, xconst,
                           parreg, parcomp, polity2)
"""

import io
import os
import sys
import urllib.request
import pandas as pd

DATA = os.path.dirname(os.path.abspath(__file__))
FH_OUT = os.path.join(DATA, "fh_subscores.csv")
POL_OUT = os.path.join(DATA, "polity5.csv")

FH_URLS = [
    "https://freedomhouse.org/sites/default/files/2025-02/All_data_FIW_2013-2024.xlsx",
    "https://freedomhouse.org/sites/default/files/2024-02/All_data_FIW_2013-2023.xlsx",
]
POLITY_URLS = [
    "https://www.systemicpeace.org/inscr/p5v2018.xls",
    "http://www.systemicpeace.org/inscr/p5v2018.xls",
]

FH_SUBQS = [f"{g}{i}" for g, n in [("A", 3), ("B", 4), ("C", 3), ("D", 4),
                                    ("E", 3), ("F", 4), ("G", 4)]
            for i in range(1, n + 1)]  # A1..A3, B1..B4, ..., G1..G4 = 25
POLITY_COMPONENTS = ["xrcomp", "xropen", "xconst", "parreg", "parcomp"]


def _fetch(urls, headers):
    for url in urls:
        try:
            print(f"  trying {url[:80]} ...")
            req = urllib.request.Request(url, headers=headers)
            with urllib.request.urlopen(req, timeout=120) as resp:
                return resp.read(), url
        except Exception as e:
            print(f"    failed: {type(e).__name__}: {str(e)[:120]}")
    return None, None


def download_fh():
    if os.path.exists(FH_OUT):
        print(f"FH already present: {FH_OUT}")
        return
    headers = {"User-Agent": "Mozilla/5.0 AIM4D-research"}
    data, url = _fetch(FH_URLS, headers)
    if data is None:
        print("MANUAL: download FH 'All Data' xlsx from "
              "https://freedomhouse.org/report/freedom-world and place at "
              f"{FH_OUT.replace('.csv', '.xlsx')}")
        return
    # FH 'All Data' is multi-sheet; the per-country sheet is usually 'FIW13-24' or similar
    xls = pd.ExcelFile(io.BytesIO(data))
    # pick the sheet with the most rows (the country-year data sheet)
    best_sheet, best_n = None, 0
    for sh in xls.sheet_names:
        tmp = pd.read_excel(xls, sheet_name=sh, header=1, nrows=5)
        if tmp.shape[1] > best_n:
            best_sheet, best_n = sh, tmp.shape[1]
    df = pd.read_excel(xls, sheet_name=best_sheet, header=1)
    # Standard FH columns: 'Country/Territory', 'Edition' (= year), plus A1..G4
    df.columns = [str(c).strip() for c in df.columns]
    country_col = next((c for c in df.columns if "Country" in c), df.columns[0])
    year_col = next((c for c in df.columns if c in ("Edition", "Year")), None)
    keep = [country_col, year_col] + [c for c in FH_SUBQS if c in df.columns]
    out = df[keep].rename(columns={country_col: "country", year_col: "year"})
    out.to_csv(FH_OUT, index=False)
    print(f"  saved FH: {len(out)} rows, {len([c for c in FH_SUBQS if c in out.columns])} subquestions")


def download_polity():
    if os.path.exists(POL_OUT):
        print(f"Polity5 already present: {POL_OUT}")
        return
    headers = {"User-Agent": "Mozilla/5.0 AIM4D-research"}
    data, url = _fetch(POLITY_URLS, headers)
    if data is None:
        print("MANUAL: download p5v2018.xls from "
              "https://www.systemicpeace.org/inscrdata.html and place at "
              f"{POL_OUT.replace('.csv', '.xls')}")
        return
    df = pd.read_excel(io.BytesIO(data))
    keep = ["ccode", "country", "year"] + [c for c in POLITY_COMPONENTS if c in df.columns]
    if "polity2" in df.columns:
        keep.append("polity2")
    out = df[keep]
    out.to_csv(POL_OUT, index=False)
    print(f"  saved Polity5: {len(out)} rows, components "
          f"{[c for c in POLITY_COMPONENTS if c in out.columns]}")


def main():
    print("Downloading Freedom House...")
    download_fh()
    print("Downloading Polity5...")
    download_polity()


if __name__ == "__main__":
    main()
