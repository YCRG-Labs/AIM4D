"""
Seed stability of the total-variation network measure.

The published logit-ratio index was reported across ten Stage 4 seeds in an
appendix. That index is discarded, so its stability table no longer describes
any quantity the paper uses. This retrains Stage 4 across seeds and records the
total variation and signed direction instead, which is what the revised network
section reports.

Outputs robustness/netdep_tv_seed_sweep.csv and its summary.
"""

import argparse
import os
import sys

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from joblib import Parallel, delayed

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from stage4_nscm.estimate import (
    BETA_COLS, FACTOR_COLS, STATE_COLS,
    build_spatial_edges, build_spatiotemporal_graph, load_all_data, train_model,
)
from contagion_seed_sweep import FOCUS

K = 5


def _one(s, x, y, edge_index, full_ei, mask_train, mask_test, in_dim,
         node_country, node_year, name_map, target_year):
    torch.set_num_threads(max(1, int(os.environ.get("AIM4D_SWEEP_THREADS", "1"))))
    model = train_model(x, y, edge_index, mask_train, mask_test, in_dim, seed=s)
    model.eval()
    with torch.no_grad():
        h_full, h_ego = model.encode(x, full_ei)
        Pf = F.softmax(model.outcome_logits(h_full), dim=-1).numpy()
        Pl = F.softmax(model.local_logits(h_ego), dim=-1).numpy()

    tv = 0.5 * np.abs(Pf - Pl).sum(axis=1)
    lvl = np.arange(K, dtype=float)
    signed = (Pf * lvl).sum(axis=1) - (Pl * lvl).sum(axis=1)

    rows = []
    for nid in range(len(node_country)):
        if node_year[nid] != target_year:
            continue
        cn = name_map.get(node_country[nid], node_country[nid])
        if cn not in FOCUS:
            continue
        rows.append({"seed": s, "country": cn,
                     "tv": float(tv[nid]), "signed": float(signed[nid])})
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=10)
    ap.add_argument("--target-year", type=int, default=2025)
    a = ap.parse_args()

    df, mapping = load_all_data()
    feature_cols = FACTOR_COLS + BETA_COLS + ["gdp_pc", "urbanization"]
    years_use = [yr for yr in sorted(df["year"].unique()) if yr >= 1990]
    complete = df.groupby("country_text_id").apply(
        lambda g: g[g["year"].isin(years_use)].dropna(subset=feature_cols + STATE_COLS)["year"].nunique()
    )
    iso3 = sorted(complete[complete >= len(years_use) * 0.8].index.tolist())
    contig, alliance, cultural = build_spatial_edges(mapping, iso3)
    (x, y, edge_index, spatial_ei, temporal_ei, mask_train, mask_test,
     node_country, node_year, N, T) = build_spatiotemporal_graph(
        df, iso3, years_use, contig, alliance, feature_cols, cultural_pairs=cultural)
    full_ei = torch.cat([spatial_ei, temporal_ei], dim=1)
    name_map = df.drop_duplicates("country_text_id").set_index("country_text_id")["country_name"].to_dict()

    n_jobs = int(os.environ.get("AIM4D_SWEEP_JOBS",
                                str(min(a.seeds, max(1, (os.cpu_count() or 4) - 1)))))
    out = [r for rows in Parallel(n_jobs=n_jobs, backend="loky")(
        delayed(_one)(s, x, y, edge_index, full_ei, mask_train, mask_test,
                      x.shape[1], node_country, node_year, name_map, a.target_year)
        for s in range(a.seeds)) for r in rows]

    d = pd.DataFrame(out)
    d.to_csv(os.path.join(os.path.dirname(__file__), "netdep_tv_seed_sweep.csv"), index=False)
    s = (d.groupby("country")[["tv", "signed"]]
           .agg(["mean", "std", "min", "max"]).round(4)
           .sort_values(("tv", "mean"), ascending=False))
    s.to_csv(os.path.join(os.path.dirname(__file__), "netdep_tv_seed_sweep_summary.csv"))
    print(s.to_string())


if __name__ == "__main__":
    main()
