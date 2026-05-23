"""
Network definition robustness (contiguity-only, alliance-only, trade-only, KNN).

Tests whether the contagion decomposition and downstream predictions are
sensitive to the choice of spatial weight matrix W.

Methodological basis:
  - LeSage & Pace (2009): sensitivity to W in spatial econometrics
  - Neumayer & Plumper (2016): W specification in political science
  - Anselin (1988): W should be theory-driven, not data-driven
  - Corrado & Fingleton (2012): coefficient stability across W

Reports:
  - MSE under each W definition
  - Spearman rank correlation of country contagion scores across W
  - Top contagion countries under each W
  - Network ablation improvement under each W
"""

import sys
import os
import warnings
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from scipy.stats import spearmanr

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from stage4_nscm.estimate import (
    load_all_data, build_spatial_edges, build_spatiotemporal_graph,
    INETARNet, train_model, network_ablation_test,
    FACTOR_COLS, BETA_COLS, STATE_COLS, TREATMENT_DIM, OUTCOME_DIM,
    HIDDEN_DIM, EPOCHS, LR, TRAIN_CUTOFF,
    neighbor_mean,
)

OUTPUT_DIR = os.path.dirname(os.path.abspath(__file__))


def build_graph_single_edge_type(df, countries_iso3, years, contig_pairs,
                                  alliance_by_year, feature_cols, edge_type):
    """
    Build spatio-temporal graph using only one edge type.
    edge_type: 'contiguity', 'alliance', 'trade', 'knn5', or 'full'
    """
    N = len(countries_iso3)
    T = len(years)
    total_nodes = N * T

    node_features = torch.zeros(total_nodes, len(feature_cols))
    node_outcomes = torch.zeros(total_nodes, OUTCOME_DIM)
    node_country = []
    node_year = []
    node_mask_train = torch.zeros(total_nodes, dtype=torch.bool)
    node_mask_test = torch.zeros(total_nodes, dtype=torch.bool)

    for t, year in enumerate(years):
        df_year = df[df["year"] == year]
        for i, iso3 in enumerate(countries_iso3):
            nid = t * N + i
            row = df_year[df_year["country_text_id"] == iso3]
            if len(row) > 0:
                node_features[nid] = torch.tensor(row[feature_cols].values[0], dtype=torch.float32)
                node_outcomes[nid] = torch.tensor(row[STATE_COLS].values[0], dtype=torch.float32)
            node_country.append(iso3)
            node_year.append(year)
            if year <= TRAIN_CUTOFF:
                node_mask_train[nid] = True
            else:
                node_mask_test[nid] = True

    feat_mean = node_features[node_mask_train].mean(dim=0)
    feat_std = node_features[node_mask_train].std(dim=0).clamp(min=1e-6)
    node_features = (node_features - feat_mean) / feat_std

    treatment = node_features[:, :TREATMENT_DIM]

    spatial_src, spatial_dst = [], []
    temporal_src, temporal_dst = [], []

    for t, year in enumerate(years):
        offset = t * N

        if edge_type in ('contiguity', 'full'):
            for (i, j) in contig_pairs:
                spatial_src.append(offset + i)
                spatial_dst.append(offset + j)

        if edge_type in ('alliance', 'full'):
            ally_pairs = alliance_by_year.get(year, set())
            for (i, j) in ally_pairs:
                spatial_src.append(offset + i)
                spatial_dst.append(offset + j)

        if edge_type in ('trade', 'knn5', 'full'):
            gdp_vals = node_features[offset:offset + N, -2].numpy()
            log_gdp = np.log1p(np.abs(gdp_vals))
            diffs = np.abs(log_gdp[:, None] - log_gdp[None, :])
            np.fill_diagonal(diffs, np.inf)
            k = 5
            for i in range(N):
                neighbors = np.argsort(diffs[i])[:k]
                for j in neighbors:
                    spatial_src.append(offset + i)
                    spatial_dst.append(offset + int(j))

        if t > 0:
            prev_offset = (t - 1) * N
            for i in range(N):
                temporal_src.append(prev_offset + i)
                temporal_dst.append(offset + i)
                temporal_src.append(offset + i)
                temporal_dst.append(prev_offset + i)

    spatial_lag = torch.zeros(total_nodes, TREATMENT_DIM)
    for t, year in enumerate(years):
        offset = t * N
        prev_offset = (t - 1) * N if t > 0 else offset
        treat_lag = treatment[prev_offset:prev_offset + N]

        step_src = [s - offset for s in spatial_src if offset <= s < offset + N]
        step_dst = [d - offset for d in spatial_dst if offset <= d < offset + N]
        spatial_lag[offset:offset + N] = neighbor_mean(treat_lag, step_src, step_dst, N)

    node_features_aug = torch.cat([
        node_features, spatial_lag, spatial_lag, spatial_lag,
    ], dim=-1)

    all_src = spatial_src + temporal_src
    all_dst = spatial_dst + temporal_dst
    edge_index = torch.tensor([all_src, all_dst], dtype=torch.long) if all_src else torch.zeros(2, 0, dtype=torch.long)
    spatial_ei = torch.tensor([spatial_src, spatial_dst], dtype=torch.long) if spatial_src else torch.zeros(2, 0, dtype=torch.long)
    temporal_ei = torch.tensor([temporal_src, temporal_dst], dtype=torch.long) if temporal_src else torch.zeros(2, 0, dtype=torch.long)

    print(f"    {edge_type}: {total_nodes} nodes, {len(spatial_src)} spatial edges, {len(temporal_src)} temporal edges")

    return (node_features_aug, node_outcomes, edge_index, spatial_ei, temporal_ei,
            node_mask_train, node_mask_test, node_country, node_year, N, T)


def compute_contagion_scores(model, x, y, edge_index, spatial_ei, node_country, node_year):
    """Compute per-country contagion scores."""
    model.eval()
    with torch.no_grad():
        full_ei = torch.cat([spatial_ei, torch.zeros(2, 0, dtype=torch.long)], dim=1) if spatial_ei.shape[1] > 0 else spatial_ei
        y_full, domestic, spillover = model.counterfactual_decompose(x, edge_index, spatial_ei)

    rows = []
    for nid in range(len(node_country)):
        spill_mag = spillover[nid].abs().sum().item()
        dom_mag = domestic[nid].abs().sum().item()
        total = spill_mag + dom_mag + 1e-10
        rows.append({
            "country_text_id": node_country[nid],
            "year": node_year[nid],
            "contagion_score": spill_mag / total,
        })

    scores = pd.DataFrame(rows)
    latest = scores[scores["year"] >= scores["year"].max() - 4]
    country_avg = latest.groupby("country_text_id")["contagion_score"].mean()
    return country_avg, scores


def run_network_variants():
    print("=" * 70)
    print("ROBUSTNESS CHECK: Network Definition Variants")
    print("=" * 70)
    print()
    print("Methodological basis:")
    print("  LeSage & Pace (2009) W sensitivity; Neumayer & Plumper (2016) W in polsci;")
    print("  Anselin (1988) theory-driven W")
    print()

    df, mapping = load_all_data()
    feature_cols = FACTOR_COLS + BETA_COLS + ["gdp_pc", "urbanization"]

    years_all = sorted(df["year"].unique())
    years_use = [y for y in years_all if y >= 1990]

    complete = df.groupby("country_text_id").apply(
        lambda g: g[g["year"].isin(years_use)].dropna(subset=feature_cols + STATE_COLS)["year"].nunique()
    )
    countries_iso3 = sorted(complete[complete >= len(years_use) * 0.8].index.tolist())
    print(f"Countries: {len(countries_iso3)}, Years: {years_use[0]}-{years_use[-1]}")

    contig_pairs, alliance_by_year, _cultural_pairs = build_spatial_edges(mapping, countries_iso3)

    edge_types = ["contiguity", "alliance", "trade", "full"]
    all_country_scores = {}
    all_results = []

    for etype in edge_types:
        print(f"\n{'='*50}")
        print(f"Edge type: {etype}")
        print(f"{'='*50}")

        print(f"\n  Building graph...")
        (x, y, edge_index, spatial_ei, temporal_ei,
         mask_train, mask_test, node_country, node_year, N, T) = \
            build_graph_single_edge_type(
                df, countries_iso3, years_use, contig_pairs, alliance_by_year,
                feature_cols, etype
            )

        in_dim = x.shape[1]
        print(f"  Training INE-TARNet...")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model = train_model(x, y, edge_index, mask_train, mask_test, in_dim)

        print(f"  Running ablation test...")
        ablation = network_ablation_test(model, x, y, edge_index, spatial_ei, temporal_ei,
                                         mask_train, mask_test, in_dim)
        print(f"    MSE full: {ablation['mse_full']:.6f}")
        print(f"    MSE no network: {ablation['mse_no_network']:.6f}")
        print(f"    Network improvement: {ablation['improvement_total_network']:.1f}%")

        print(f"  Computing contagion scores...")
        country_avg, full_scores = compute_contagion_scores(
            model, x, y, edge_index, spatial_ei, node_country, node_year
        )
        all_country_scores[etype] = country_avg

        top10 = country_avg.sort_values(ascending=False).head(10)
        print(f"\n  Top 10 network-influenced countries:")
        for iso3, score in top10.items():
            print(f"    {iso3}: {score:.3f}")

        all_results.append({
            "edge_type": etype,
            "mse_full": ablation["mse_full"],
            "mse_no_network": ablation["mse_no_network"],
            "network_improvement_pct": ablation["improvement_total_network"],
            "n_spatial_edges": spatial_ei.shape[1],
            "mean_contagion": country_avg.mean(),
            "std_contagion": country_avg.std(),
        })

    print(f"\n{'='*50}")
    print("Spearman Rank Correlations of Country Contagion Scores")
    print(f"{'='*50}")

    etypes_with_scores = [e for e in edge_types if e in all_country_scores]
    corr_matrix = pd.DataFrame(index=etypes_with_scores, columns=etypes_with_scores, dtype=float)

    for e1 in etypes_with_scores:
        for e2 in etypes_with_scores:
            shared = all_country_scores[e1].index.intersection(all_country_scores[e2].index)
            if len(shared) < 5:
                corr_matrix.loc[e1, e2] = np.nan
                continue
            rho, p = spearmanr(
                all_country_scores[e1].loc[shared],
                all_country_scores[e2].loc[shared],
            )
            corr_matrix.loc[e1, e2] = rho

    print(corr_matrix.to_string(float_format="%.3f"))

    off_diag = []
    for i, e1 in enumerate(etypes_with_scores):
        for j, e2 in enumerate(etypes_with_scores):
            if i < j:
                val = corr_matrix.loc[e1, e2]
                if not np.isnan(val):
                    off_diag.append(val)

    if off_diag:
        mean_rho = np.mean(off_diag)
        min_rho = np.min(off_diag)
        print(f"\n  Mean pairwise rank correlation: {mean_rho:.3f}")
        print(f"  Min pairwise rank correlation: {min_rho:.3f}")
        print(f"  {'STABLE (>.85)' if min_rho > 0.85 else 'MODERATE (.70-.85)' if min_rho > 0.70 else 'SENSITIVE (<.70)'}")

    print(f"\n{'='*50}")
    print("SUMMARY TABLE")
    print(f"{'='*50}")
    summary = pd.DataFrame(all_results)
    print(summary.to_string(index=False, float_format="%.4f"))

    summary.to_csv(os.path.join(OUTPUT_DIR, "network_variants_results.csv"), index=False)
    corr_matrix.to_csv(os.path.join(OUTPUT_DIR, "network_rank_correlations.csv"))

    per_country = pd.DataFrame(all_country_scores)
    per_country.index.name = "country_text_id"
    per_country.to_csv(os.path.join(OUTPUT_DIR, "contagion_by_edge_type.csv"))

    print(f"\nSaved to robustness/network_variants_results.csv")
    print(f"Saved to robustness/network_rank_correlations.csv")
    print(f"Saved to robustness/contagion_by_edge_type.csv (per-country contagion by edge type)")

    return summary, corr_matrix


if __name__ == "__main__":
    run_network_variants()
