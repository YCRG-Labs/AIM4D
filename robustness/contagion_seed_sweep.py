"""
Contagion decomposition seed sweep.

Retrains the Stage 4 INE-TARNet across N random seeds and captures, for each
seed, the 2025 contagion share for a fixed set of focus countries (Hungary,
Türkiye, Poland, USA, Denmark, Ukraine, Serbia, Tunisia, Brazil, Argentina).
Writes mean ± std across seeds so the paper's case-study percentages (e.g.,
Hungary 68%, Türkiye 71% domestic) can be reported with a stability range.

Outputs:
  robustness/contagion_seed_sweep.csv          per-seed per-country contagion
  robustness/contagion_seed_sweep_summary.csv  mean / std / min / max
"""

import argparse
import os
import sys

import numpy as np
import pandas as pd
import torch
from joblib import Parallel, delayed

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from stage4_nscm.estimate import (
    BETA_COLS, FACTOR_COLS, STATE_COLS, OUTCOME_DIM, TREATMENT_DIM,
    build_spatial_edges, build_spatiotemporal_graph,
    load_all_data, train_model,
)


def _sweep_one(s, x, y, edge_index, full_ei, spatial_ei, mask_train, mask_test,
               in_dim, node_country, node_year, name_map, target_year):
    torch.set_num_threads(max(1, int(os.environ.get("AIM4D_SWEEP_THREADS", "1"))))
    model = train_model(x, y, edge_index, mask_train, mask_test, in_dim, seed=s)

    with torch.no_grad():
        _, domestic, spillover = model.counterfactual_decompose(x, full_ei, spatial_ei)

    rows = []
    for nid in range(len(node_country)):
        if node_year[nid] != target_year:
            continue
        country_name = name_map.get(node_country[nid], node_country[nid])
        if country_name not in FOCUS:
            continue
        spill_mag = spillover[nid].abs().sum().item()
        dom_mag = domestic[nid].abs().sum().item()
        total = spill_mag + dom_mag + 1e-10
        contagion = spill_mag / total
        rows.append({
            "seed": s,
            "country": country_name,
            "contagion": contagion,
            "domestic": 1 - contagion,
        })
    return rows


FOCUS = [
    "Hungary", "Türkiye", "Poland", "United States of America", "Denmark",
    "Ukraine", "Serbia", "Tunisia", "Brazil", "Argentina",
]


def contagion_for_year(model, x, full_ei, node_country, node_year, target_year, name_map):
    model.eval()
    with torch.no_grad():
        _, _, spillover = model.counterfactual_decompose(x, full_ei, full_ei)
        h_full, h_ego = model.encode(x, full_ei)
        domestic = model.outcome_logits(h_ego) - model.outcome_logits(h_full)

    rows = []
    for nid in range(len(node_country)):
        if node_year[nid] != target_year:
            continue
        spill_mag = spillover[nid].abs().sum().item()
        dom_mag = abs(domestic[nid].abs().sum().item())
        total = spill_mag + dom_mag + 1e-10
        country_name = name_map.get(node_country[nid], node_country[nid])
        rows.append({"country_name": country_name, "contagion": spill_mag / total})
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, default=10)
    parser.add_argument("--target-year", type=int, default=2025)
    args = parser.parse_args()

    df, mapping = load_all_data()
    feature_cols = FACTOR_COLS + BETA_COLS + ["gdp_pc", "urbanization"]
    years_use = [y for y in sorted(df["year"].unique()) if y >= 1990]
    complete = df.groupby("country_text_id").apply(
        lambda g: g[g["year"].isin(years_use)].dropna(subset=feature_cols + STATE_COLS)["year"].nunique()
    )
    countries_iso3 = sorted(complete[complete >= len(years_use) * 0.8].index.tolist())

    contig_pairs, alliance_by_year, cultural_pairs = build_spatial_edges(mapping, countries_iso3)
    (x, y, edge_index, spatial_ei, temporal_ei,
     mask_train, mask_test, node_country, node_year, N, T) = build_spatiotemporal_graph(
        df, countries_iso3, years_use, contig_pairs, alliance_by_year, feature_cols,
        cultural_pairs=cultural_pairs,
    )
    in_dim = x.shape[1]
    full_ei = torch.cat([spatial_ei, temporal_ei], dim=1)
    name_map = df.drop_duplicates("country_text_id").set_index("country_text_id")["country_name"].to_dict()

    n_jobs = int(os.environ.get("AIM4D_SWEEP_JOBS",
                                str(min(args.seeds, max(1, (os.cpu_count() or 4) - 1)))))
    seed_rows = Parallel(n_jobs=n_jobs, backend="loky")(
        delayed(_sweep_one)(s, x, y, edge_index, full_ei, spatial_ei,
                            mask_train, mask_test, in_dim, node_country,
                            node_year, name_map, args.target_year)
        for s in range(args.seeds)
    )

    all_rows = [r for rows in seed_rows for r in rows]
    for row in all_rows:
        print(f"  seed {row['seed']} {row['country']:<30} "
              f"contagion={row['contagion']:.3f}  domestic={row['domestic']:.3f}")

    out = pd.DataFrame(all_rows)
    out_path = os.path.join(os.path.dirname(__file__), "contagion_seed_sweep.csv")
    out.to_csv(out_path, index=False)

    summary = (out.groupby("country")["contagion"]
                  .agg(["mean", "std", "min", "max"])
                  .reset_index()
                  .sort_values("mean", ascending=False))
    summary_path = os.path.join(os.path.dirname(__file__), "contagion_seed_sweep_summary.csv")
    summary.to_csv(summary_path, index=False)

    print(f"\n=== Summary across {args.seeds} seeds (target year {args.target_year}) ===")
    print(summary.round(4).to_string(index=False))
    print(f"\nWrote {out_path}\nWrote {summary_path}")


if __name__ == "__main__":
    main()
