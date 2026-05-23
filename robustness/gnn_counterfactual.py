"""
GNN counterfactual decomposition — neighbor-swap analysis.

INETARNet's outcome head depends on (a) the ego country's own features and (b)
the spatial lag block aggregated from its neighbors. We exploit that to
compute counterfactuals of the form:

  "What would Hungary's predicted trajectory look like if it had Türkiye's
   neighbor profile in 2024 instead of its own?"

We refit Stage 4's INETARNet once, then for each (target, reference) pair at
each year T, we predict three quantities:

  - actual:        target's ego  + target's spatial lag         (baseline)
  - swap:          target's ego  + reference's spatial lag      (counterfactual)
  - no_contagion:  target's ego  + zeroed spatial lag           (existing method)

Outputs help anchor causal-style claims like "without alliance/contiguity
contagion, Hungary would have remained ~X polyarchy". The delta is **descriptive
under the model's identifying assumptions**, not a randomized-experiment
counterfactual.

Output: robustness/gnn_counterfactual.csv with one row per (target, reference,
year, outcome_class). Run time ~5-10 min on Brev.
"""

import os
import sys
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
from stage4_nscm.estimate import (
    load_all_data, build_spatial_edges, build_spatiotemporal_graph,
    train_model, INETARNet, STATE_COLS, TREATMENT_DIM,
    FACTOR_COLS, BETA_COLS,
)

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "gnn_counterfactual.csv")

PAIRS = [
    ("Hungary",                  "Türkiye"),
    ("Hungary",                  "Denmark"),
    ("Hungary",                  "Russia"),
    ("Türkiye",                  "Denmark"),
    ("Türkiye",                  "Hungary"),
    ("Poland",                   "Hungary"),
    ("Brazil",                   "United States of America"),
    ("Brazil",                   "Bolivia"),
    ("United States of America", "Denmark"),
    ("United States of America", "Hungary"),
]

CF_YEARS = [2017, 2019, 2021, 2023, 2025]


def main():
    print("Loading Stage 4 data...")
    df, mapping = load_all_data()
    feature_cols = FACTOR_COLS + BETA_COLS + ["gdp_pc", "urbanization"]

    years_all = sorted(df["year"].unique())
    years_use = [y for y in years_all if y >= 1990]
    complete = df.groupby("country_text_id").apply(
        lambda g: g[g["year"].isin(years_use)].dropna(subset=feature_cols + STATE_COLS)["year"].nunique()
    )
    countries_iso3 = sorted(complete[complete >= len(years_use) * 0.8].index.tolist())
    print(f"Countries: {len(countries_iso3)}, Years: {years_use[0]}-{years_use[-1]}")

    contig_pairs, alliance_by_year, cultural_pairs = build_spatial_edges(mapping, countries_iso3)
    (x, y, edge_index, spatial_ei, temporal_ei,
     mask_train, mask_test, node_country, node_year, N, T) = build_spatiotemporal_graph(
        df, countries_iso3, years_use, contig_pairs, alliance_by_year, feature_cols,
        cultural_pairs=cultural_pairs,
    )
    in_dim = x.shape[1]

    print(f"\nTraining INE-TARNet for counterfactual analysis...")
    model = train_model(x, y, edge_index, mask_train, mask_test, in_dim, seed=42)

    name_to_iso = df.drop_duplicates("country_text_id").set_index("country_name")["country_text_id"].to_dict()
    iso_to_idx = {iso: i for i, iso in enumerate(countries_iso3)}
    year_to_idx = {yr: t for t, yr in enumerate(years_use)}

    def node_id(country_name, year):
        iso = name_to_iso.get(country_name)
        if iso is None or iso not in iso_to_idx or year not in year_to_idx:
            return None
        return year_to_idx[year] * N + iso_to_idx[iso]

    spatial_lag_dim = model.spatial_lag_dim

    rows = []
    model.eval()
    with torch.no_grad():
        h_full, _ = model.encode(x, edge_index)
        y_full_actual = F.softmax(model.outcome_logits(h_full), dim=-1).numpy()

        x_zero = x.clone()
        x_zero[:, -spatial_lag_dim:] = 0.0
        h_zero, _ = model.encode(x_zero, torch.zeros(2, 0, dtype=torch.long))
        y_no_contagion = F.softmax(model.outcome_logits(h_zero), dim=-1).numpy()

        for target_name, ref_name in PAIRS:
            for year in CF_YEARS:
                tgt_id = node_id(target_name, year)
                ref_id = node_id(ref_name, year)
                if tgt_id is None or ref_id is None:
                    continue
                x_swap = x.clone()
                x_swap[tgt_id, -spatial_lag_dim:] = x[ref_id, -spatial_lag_dim:]
                h_swap, _ = model.encode(x_swap, edge_index)
                y_swap = F.softmax(model.outcome_logits(h_swap), dim=-1).numpy()

                actual_row = y_full_actual[tgt_id]
                swap_row = y_swap[tgt_id]
                noctg_row = y_no_contagion[tgt_id]

                for k, state in enumerate(STATE_COLS):
                    rows.append({
                        "target": target_name,
                        "reference": ref_name,
                        "year": year,
                        "outcome_state": state,
                        "actual_prob": float(actual_row[k]),
                        "swap_prob": float(swap_row[k]),
                        "no_contagion_prob": float(noctg_row[k]),
                        "delta_swap": float(swap_row[k] - actual_row[k]),
                        "delta_no_contagion": float(noctg_row[k] - actual_row[k]),
                    })

    df_out = pd.DataFrame(rows)
    df_out.to_csv(OUT, index=False)

    print("\n" + "=" * 78)
    print("GNN COUNTERFACTUAL: NEIGHBOR-SWAP DECOMPOSITION")
    print("=" * 78)
    summary = (df_out.groupby(["target", "reference"])
               .agg(mean_abs_delta_swap=("delta_swap", lambda s: float(np.mean(np.abs(s)))),
                    mean_abs_delta_no_contagion=("delta_no_contagion", lambda s: float(np.mean(np.abs(s)))),
                    n_year_state_obs=("delta_swap", "count"))
               .reset_index()
               .sort_values("mean_abs_delta_swap", ascending=False))
    print(summary.to_string(index=False))

    libdem_rows = df_out[df_out["outcome_state"] == "liberal_democracy"]
    print(f"\nLiberal-democracy P(.) under neighbor swap (year 2025):")
    swap_2025 = libdem_rows[libdem_rows["year"] == 2025]
    for _, r in swap_2025.iterrows():
        print(f"  {r['target']:25s} | actual={r['actual_prob']:.3f}  "
              f"swap-to-{r['reference']}={r['swap_prob']:.3f}  "
              f"(Δ={r['delta_swap']:+.3f})")

    print(f"\nWrote {OUT}")


if __name__ == "__main__":
    main()
