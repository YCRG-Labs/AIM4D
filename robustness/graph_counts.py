"""Node and edge counts of the Stage 4 spatio-temporal graph, as reported in the supplement.

Rebuilds the graph with Stage 4's own construction code and no training, so the counts in
the manuscript are re-derived rather than transcribed from a run log. Temporal edges run
past to present only, so their count is (countries) x (year transitions).

Output: robustness/graph_counts.csv.
"""

import os
import sys

import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from stage4_nscm.estimate import (BETA_COLS, FACTOR_COLS, STATE_COLS, build_spatial_edges,
                                  build_spatiotemporal_graph, load_all_data)

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "graph_counts.csv")


def main():
    df, mapping = load_all_data()
    feature_cols = FACTOR_COLS + BETA_COLS + ["gdp_pc", "urbanization"]
    years_use = [y for y in sorted(df["year"].unique()) if y >= 1990]
    complete = df.groupby("country_text_id").apply(
        lambda g: g[g["year"].isin(years_use)].dropna(subset=feature_cols + STATE_COLS)["year"].nunique()
    )
    countries = sorted(complete[complete >= len(years_use) * 0.8].index.tolist())
    contig_pairs, alliance_by_year, cultural_pairs = build_spatial_edges(mapping, countries)
    built = build_spatiotemporal_graph(df, countries, years_use, contig_pairs, alliance_by_year,
                                       feature_cols, cultural_pairs=cultural_pairs)
    x, edge_index, spatial_ei, temporal_ei = built[0], built[2], built[3], built[4]
    row = {"countries": len(countries), "years": len(years_use),
           "year_first": years_use[0], "year_last": years_use[-1],
           "nodes": int(x.shape[0]), "edges_total": int(edge_index.shape[1]),
           "edges_spatial": int(spatial_ei.shape[1]), "edges_temporal": int(temporal_ei.shape[1])}
    row["temporal_is_forward_only"] = int(row["edges_temporal"] == len(countries) * (len(years_use) - 1))
    pd.DataFrame([row]).to_csv(OUT, index=False)
    print(pd.DataFrame([row]).to_string(index=False))
    print(f"\nWrote {OUT}")


if __name__ == "__main__":
    main()
