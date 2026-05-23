"""
F6: UCDP-GED conflict-onset meta-features for Stage 5.

Builds 6 country-year features from UCDP-GED v25.1 (state-based conflict
only, type_of_violence==1) following Hegre et al. (2019) ViEWS convention
and Beger-Dorff-Ward (2014) CoupCast spatial-lag specification:

  ucdp_onset_lag1        binary: state-based onset in year y-1 (>=25 BD,
                         inactive y-2 and y-3)
  ucdp_active_5y         binary: any active conflict-year in [y-5, y-1]
  ucdp_years_since_onset cumulative years since last onset; cap at 25
  ucdp_log_bd_lag1       log1p(sum best-estimate battle-deaths) in y-1
  ucdp_neighbor_onset_lag1  count of contiguous neighbors with onset in y-1
  ucdp_neighbor_log_bd_lag1 log1p(sum of neighbor BD in y-1)

ALL features are STRICTLY past-only (shift(1) on every aggregation; rolling
windows applied AFTER the shift). No same-year inclusion.

UCDP `country_id` is Gleditsch-Ward; we bridge to V-Dem `country_text_id` via
`data/cow_iso3_mapping.csv` plus a small override dict for successor states
(Sudan/South Sudan, Yugoslavia, Czechoslovakia, Germany unification).

Output: data/ucdp_features.csv with columns
  country_text_id, year, plus the 6 features above.
"""

import os
import sys
import numpy as np
import pandas as pd

DATA = os.path.dirname(os.path.abspath(__file__))
UCDP_CSV = os.path.join(DATA, "ucdp_ged.csv")
COW_MAP = os.path.join(DATA, "cow_iso3_mapping.csv")
CONTIG_CSV = os.path.join(DATA, "contiguity", "DirectContiguity320", "contdird.csv")
OUT = os.path.join(DATA, "ucdp_features.csv")

GW_OVERRIDES = {
    345: "SRB",
    347: "KOS",
    340: "SRB",
    626: "SSD",
    625: "SDN",
    679: "YEM",
    818: "VNM",
    315: "CZE",
    260: "DEU",
    265: "DEU",
}

YEAR_MIN = 1989
YEAR_MAX = 2025


def _load_ucdp_country_year():
    if not os.path.exists(UCDP_CSV):
        sys.exit(f"Missing {UCDP_CSV}. Run: python3 data/download_ucdp.py")
    df = pd.read_csv(UCDP_CSV, low_memory=False)
    print(f"Loaded UCDP-GED: {len(df)} events, {df['year'].min()}-{df['year'].max()}")

    df = df[df["type_of_violence"] == 1]
    print(f"  state-based events: {len(df)}")

    cow = pd.read_csv(COW_MAP).set_index("COWcode")["country_text_id"]
    df["country_text_id"] = df["country_id"].map(GW_OVERRIDES).fillna(
        df["country_id"].map(cow))
    unmapped = df[df["country_text_id"].isna()]
    if len(unmapped):
        print(f"  WARN: {len(unmapped)} events from {unmapped['country_id'].nunique()} unmapped GW codes")
        print(f"    sample: {unmapped['country_id'].value_counts().head().to_dict()}")
        df = df.dropna(subset=["country_text_id"])

    cy = (df.groupby(["country_text_id", "year"])["best"]
            .sum()
            .reset_index()
            .rename(columns={"best": "bd"}))
    return cy


def _expand_to_balanced_panel(cy):
    """Expand to balanced (country_text_id × year) panel covering all V-Dem countries."""
    vdem_path = os.path.join(DATA, "vdem_v16.csv")
    vdem = pd.read_csv(vdem_path, usecols=["country_text_id", "year"], low_memory=False)
    vdem = vdem[(vdem["year"] >= YEAR_MIN) & (vdem["year"] <= YEAR_MAX)]
    base = vdem.drop_duplicates()
    panel = base.merge(cy, on=["country_text_id", "year"], how="left")
    panel["bd"] = panel["bd"].fillna(0.0)
    panel = panel.sort_values(["country_text_id", "year"]).reset_index(drop=True)
    return panel


def _compute_country_features(panel):
    """Onset, active-5y, years-since-onset, log-BD with strict past-only lags."""
    panel["active"] = (panel["bd"] >= 25).astype(int)
    g = panel.groupby("country_text_id", group_keys=False)

    panel["active_lag1"] = g["active"].shift(1)
    panel["active_lag2"] = g["active"].shift(2)
    panel["active_lag3"] = g["active"].shift(3)
    panel["bd_lag1"] = g["bd"].shift(1)

    panel["ucdp_onset_lag1"] = (
        (panel["active_lag1"] == 1)
        & (panel["active_lag2"].fillna(0) == 0)
        & (panel["active_lag3"].fillna(0) == 0)
    ).astype(int)

    panel["ucdp_active_5y"] = (
        g["active"].shift(1)
        .rolling(5, min_periods=1).max()
        .reset_index(level=0, drop=True)
    ).fillna(0).astype(int)

    panel["ucdp_log_bd_lag1"] = np.log1p(panel["bd_lag1"].fillna(0.0))

    def _yso(s):
        last = -9999
        out = np.zeros(len(s), dtype=int)
        for i, v in enumerate(s.values):
            if v == 1:
                last = i
            out[i] = (i - last) if last >= 0 else 25
        return pd.Series(out, index=s.index).clip(upper=25)
    panel["ucdp_years_since_onset"] = (
        panel.groupby("country_text_id")["ucdp_onset_lag1"]
        .transform(_yso)
    )

    return panel


def _load_neighbors():
    """Load COW contiguity (DirectContiguity320). Return dict iso3 -> set(iso3) neighbors."""
    if not os.path.exists(CONTIG_CSV):
        print(f"  Contiguity file not found at {CONTIG_CSV}; skipping neighbor features")
        return None
    cont = pd.read_csv(CONTIG_CSV)
    if "conttype" in cont.columns:
        cont = cont[cont["conttype"] <= 4]

    cow = pd.read_csv(COW_MAP).set_index("COWcode")["country_text_id"]
    cont["iso_a"] = cont["state1no"].map(cow)
    cont["iso_b"] = cont["state2no"].map(cow)
    cont = cont.dropna(subset=["iso_a", "iso_b"])

    neigh = {}
    for _, r in cont.iterrows():
        neigh.setdefault(r["iso_a"], set()).add(r["iso_b"])
        neigh.setdefault(r["iso_b"], set()).add(r["iso_a"])
    return neigh


def _compute_neighbor_features(panel, neigh):
    """For each (country, year), sum onset_lag1 and log_bd_lag1 over contiguous neighbors."""
    if neigh is None:
        panel["ucdp_neighbor_onset_lag1"] = 0.0
        panel["ucdp_neighbor_log_bd_lag1"] = 0.0
        return panel

    onset_wide = panel.pivot(index="country_text_id", columns="year",
                              values="ucdp_onset_lag1").fillna(0.0)
    bd_wide = panel.pivot(index="country_text_id", columns="year",
                           values="bd_lag1").fillna(0.0)

    neigh_onset = np.zeros(len(panel), dtype=float)
    neigh_bd = np.zeros(len(panel), dtype=float)
    for i, row in enumerate(panel.itertuples(index=False)):
        ego = row.country_text_id
        yr = row.year
        ns = neigh.get(ego, set())
        if not ns:
            continue
        ns_in_panel = [n for n in ns if n in onset_wide.index and yr in onset_wide.columns]
        if not ns_in_panel:
            continue
        neigh_onset[i] = onset_wide.loc[ns_in_panel, yr].sum()
        neigh_bd[i] = bd_wide.loc[ns_in_panel, yr].sum()

    panel["ucdp_neighbor_onset_lag1"] = neigh_onset
    panel["ucdp_neighbor_log_bd_lag1"] = np.log1p(neigh_bd)
    return panel


def main():
    print("=" * 70)
    print("Building UCDP-GED Stage-5 meta-features (F6)")
    print("=" * 70)
    cy = _load_ucdp_country_year()
    panel = _expand_to_balanced_panel(cy)
    print(f"Balanced panel: {len(panel)} country-years, "
          f"{panel['country_text_id'].nunique()} countries, "
          f"{panel['year'].min()}-{panel['year'].max()}")

    panel = _compute_country_features(panel)

    print("Loading contiguity edges...")
    neigh = _load_neighbors()
    if neigh:
        print(f"  contiguity dict: {sum(len(v) for v in neigh.values())//2} undirected pairs")
    panel = _compute_neighbor_features(panel, neigh)

    feat_cols = [
        "ucdp_onset_lag1",
        "ucdp_active_5y",
        "ucdp_years_since_onset",
        "ucdp_log_bd_lag1",
        "ucdp_neighbor_onset_lag1",
        "ucdp_neighbor_log_bd_lag1",
    ]
    out = panel[["country_text_id", "year"] + feat_cols]
    out.to_csv(OUT, index=False)
    print(f"\nWrote {OUT}")
    print(f"  {len(out)} rows, {out['country_text_id'].nunique()} countries, "
          f"{out['year'].min()}-{out['year'].max()}")
    print(f"  Active conflict country-years: {int(panel['active'].sum())}")
    print(f"  Onset events: {int(panel['ucdp_onset_lag1'].sum())} "
          f"({100*panel['ucdp_onset_lag1'].mean():.2f}% base rate)")


if __name__ == "__main__":
    main()
