"""
Refits the Stage 4 INE-TARNet across N random seeds and reports the
distribution of the learned convex network weights (alpha_contig,
alpha_alliance, alpha_trade, alpha_cultural). Used to assess whether the
near-uniform weights reported in the paper are data-driven or
initialization-driven.
"""

import argparse
import os
import sys

import pandas as pd
from joblib import Parallel, delayed

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from stage4_nscm.estimate import (
    BETA_COLS, FACTOR_COLS, STATE_COLS,
    build_spatial_edges, build_spatiotemporal_graph,
    load_all_data, train_model,
)


def _sweep_one(s, x, y, edge_index, mask_train, mask_test, in_dim):
    import torch
    torch.set_num_threads(max(1, int(os.environ.get("AIM4D_SWEEP_THREADS", "1"))))
    model = train_model(x, y, edge_index, mask_train, mask_test, in_dim, seed=s)
    w = model.get_w_weights().detach().numpy()
    return {
        "seed": s,
        "alpha_contig": float(w[0]),
        "alpha_alliance": float(w[1]),
        "alpha_trade": float(w[2]),
        "alpha_cultural": float(w[3]),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, default=10)
    args = parser.parse_args()

    df, mapping = load_all_data()
    feature_cols = FACTOR_COLS + BETA_COLS + ["gdp_pc", "urbanization"]
    years_use = [y for y in sorted(df["year"].unique()) if y >= 1990]
    complete = df.groupby("country_text_id").apply(
        lambda g: g[g["year"].isin(years_use)].dropna(subset=feature_cols + STATE_COLS)["year"].nunique()
    )
    countries_iso3 = sorted(complete[complete >= len(years_use) * 0.8].index.tolist())

    contig_pairs, alliance_by_year, cultural_pairs = build_spatial_edges(mapping, countries_iso3)
    (x, y, edge_index, _spatial_ei, _temporal_ei,
     mask_train, mask_test, *_) = build_spatiotemporal_graph(
        df, countries_iso3, years_use, contig_pairs, alliance_by_year, feature_cols,
        cultural_pairs=cultural_pairs,
    )
    in_dim = x.shape[1]

    n_jobs = int(os.environ.get("AIM4D_SWEEP_JOBS",
                                str(min(args.seeds, max(1, (os.cpu_count() or 4) - 1)))))
    rows = Parallel(n_jobs=n_jobs, backend="loky")(
        delayed(_sweep_one)(s, x, y, edge_index, mask_train, mask_test, in_dim)
        for s in range(args.seeds)
    )
    for r in rows:
        print(f"seed {r['seed']}: contig={r['alpha_contig']:.3f} alliance={r['alpha_alliance']:.3f} "
              f"trade={r['alpha_trade']:.3f} cultural={r['alpha_cultural']:.3f}")

    out_dir = os.path.dirname(os.path.abspath(__file__))
    out = pd.DataFrame(rows)
    out.to_csv(os.path.join(out_dir, "network_seed_sweep.csv"), index=False)

    summary = out[["alpha_contig", "alpha_alliance", "alpha_trade", "alpha_cultural"]].agg(["mean", "std", "min", "max"])
    summary.to_csv(os.path.join(out_dir, "network_seed_sweep_summary.csv"))
    print("\n=== Summary across seeds ===")
    print(summary.round(4).to_string())


if __name__ == "__main__":
    main()
