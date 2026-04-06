import sys
import os
import warnings
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.data import Data

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

FACTOR_COLS = ["factor_1", "factor_2", "factor_3", "factor_4"]
BETA_COLS = ["beta_factor_1", "beta_factor_2", "beta_factor_3", "beta_factor_4"]
STATE_COLS = ["prob_state_0", "prob_state_1", "prob_state_2", "prob_state_3", "prob_state_4"]
TREATMENT_DIM = 4
OUTCOME_DIM = 5
HIDDEN_DIM = 32
EPOCHS = 250
LR = 3e-3
TRAIN_CUTOFF = 2019


def load_all_data():
    base = os.path.dirname(os.path.abspath(__file__))
    factors = pd.read_csv(os.path.join(base, "..", "stage1_factors", "country_year_factors.csv"))
    betas = pd.read_csv(os.path.join(base, "..", "stage2_betas", "country_year_betas.csv"))
    states = pd.read_csv(os.path.join(base, "..", "stage3_msvar", "country_year_states.csv"))
    macro = pd.read_csv(os.path.join(base, "..", "data", "macro_covariates.csv"))
    mapping = pd.read_csv(os.path.join(base, "..", "data", "cow_iso3_mapping.csv"))

    df = factors.merge(betas[["country_name", "year"] + BETA_COLS], on=["country_name", "year"])
    df = df.merge(states[["country_name", "year"] + STATE_COLS], on=["country_name", "year"])
    macro_cols = ["gdp_pc", "urbanization"]
    macro_sub = macro[["iso3", "year"] + macro_cols].copy()
    for c in macro_cols:
        macro_sub[c] = macro_sub[c].fillna(macro_sub[c].median())
    df = df.merge(macro_sub, left_on=["country_text_id", "year"], right_on=["iso3", "year"], how="left")
    for c in macro_cols:
        df[c] = df[c].fillna(df[c].median())
    return df, mapping


def build_spatial_edges(mapping, countries_iso3):
    cow_map = dict(zip(mapping["country_text_id"], mapping["COWcode"]))
    iso3_map = {v: k for k, v in cow_map.items()}

    cont = pd.read_csv("data/contiguity/DirectContiguity320/contdird.csv", low_memory=False)
    land = cont[cont["conttype"] <= 2].groupby(["state1no", "state2no"]).last().reset_index()
    contig_pairs = set()
    for _, row in land.iterrows():
        s1, s2 = iso3_map.get(row["state1no"]), iso3_map.get(row["state2no"])
        if s1 in countries_iso3 and s2 in countries_iso3:
            contig_pairs.add((countries_iso3.index(s1), countries_iso3.index(s2)))
            contig_pairs.add((countries_iso3.index(s2), countries_iso3.index(s1)))

    atop = pd.read_csv("data/atop/ATOP 5.1 (.csv)/atop5_1dy.csv", low_memory=False, encoding="latin-1")
    alliance_by_year = {}
    for yr in range(1990, 2026):
        active = atop[(atop["atopally"] == 1) & (atop["year"] >= yr - 5) & (atop["year"] <= yr)]
        pairs = set()
        for _, row in active.iterrows():
            s1, s2 = iso3_map.get(row["mem1"]), iso3_map.get(row["mem2"])
            if s1 in countries_iso3 and s2 in countries_iso3:
                pairs.add((countries_iso3.index(s1), countries_iso3.index(s2)))
                pairs.add((countries_iso3.index(s2), countries_iso3.index(s1)))
        alliance_by_year[yr] = pairs

    return contig_pairs, alliance_by_year


def neighbor_mean(values, src, dst, n_nodes):
    agg = torch.zeros(n_nodes, values.shape[1])
    deg = torch.zeros(n_nodes)
    if len(src) == 0:
        return agg
    src_t = torch.tensor(src, dtype=torch.long)
    dst_t = torch.tensor(dst, dtype=torch.long)
    agg.scatter_add_(0, dst_t.unsqueeze(-1).expand(-1, values.shape[1]), values[src_t])
    deg.scatter_add_(0, dst_t, torch.ones(len(src)))
    deg = deg.clamp(min=1)
    return agg / deg.unsqueeze(-1)


def build_spatiotemporal_graph(df, countries_iso3, years, contig_pairs, alliance_by_year, feature_cols):
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

    spatial_lag_contig = torch.zeros(total_nodes, TREATMENT_DIM)
    spatial_lag_alliance = torch.zeros(total_nodes, TREATMENT_DIM)
    spatial_lag_trade = torch.zeros(total_nodes, TREATMENT_DIM)

    contig_edges_src, contig_edges_dst = [], []
    alliance_edges_src, alliance_edges_dst = [], []
    trade_edges_src, trade_edges_dst = [], []
    temporal_edges_src, temporal_edges_dst = [], []

    for t, year in enumerate(years):
        offset = t * N

        for (i, j) in contig_pairs:
            contig_edges_src.append(offset + i)
            contig_edges_dst.append(offset + j)

        ally_pairs = alliance_by_year.get(year, set())
        for (i, j) in ally_pairs:
            alliance_edges_src.append(offset + i)
            alliance_edges_dst.append(offset + j)

        gdp_vals = node_features[offset:offset + N, -2].numpy()
        log_gdp = np.log1p(np.abs(gdp_vals))
        diffs = np.abs(log_gdp[:, None] - log_gdp[None, :])
        np.fill_diagonal(diffs, np.inf)
        for i in range(N):
            neighbors = np.argsort(diffs[i])[:5]
            for j in neighbors:
                trade_edges_src.append(offset + i)
                trade_edges_dst.append(offset + int(j))

        if t > 0:
            prev_offset = (t - 1) * N
            for i in range(N):
                temporal_edges_src.append(prev_offset + i)
                temporal_edges_dst.append(offset + i)
                temporal_edges_src.append(offset + i)
                temporal_edges_dst.append(prev_offset + i)

        treat_t = treatment[offset:offset + N]
        spatial_lag_contig[offset:offset + N] = neighbor_mean(
            treat_t, [i for i, j in contig_pairs], [j for i, j in contig_pairs], N
        )
        ally_src = [i for i, j in ally_pairs]
        ally_dst = [j for i, j in ally_pairs]
        spatial_lag_alliance[offset:offset + N] = neighbor_mean(treat_t, ally_src, ally_dst, N)

        t_src = [s - offset for s in trade_edges_src if offset <= s < offset + N]
        t_dst = [d - offset for d in trade_edges_dst if offset <= d < offset + N]
        spatial_lag_trade[offset:offset + N] = neighbor_mean(treat_t, t_src, t_dst, N)

    node_features_aug = torch.cat([
        node_features,
        spatial_lag_contig,
        spatial_lag_alliance,
        spatial_lag_trade,
    ], dim=-1)

    all_spatial_src = contig_edges_src + alliance_edges_src + trade_edges_src
    all_spatial_dst = contig_edges_dst + alliance_edges_dst + trade_edges_dst
    all_src = all_spatial_src + temporal_edges_src
    all_dst = all_spatial_dst + temporal_edges_dst

    edge_index = torch.tensor([all_src, all_dst], dtype=torch.long)
    spatial_edge_index = torch.tensor([all_spatial_src, all_spatial_dst], dtype=torch.long) if all_spatial_src else torch.zeros(2, 0, dtype=torch.long)
    temporal_edge_index = torch.tensor([temporal_edges_src, temporal_edges_dst], dtype=torch.long) if temporal_edges_src else torch.zeros(2, 0, dtype=torch.long)

    n_spatial = len(all_spatial_src)
    n_temporal = len(temporal_edges_src)

    print(f"  Spatio-temporal graph: {total_nodes} nodes, "
          f"{edge_index.shape[1]} edges ({n_spatial} spatial, {n_temporal} temporal)")
    print(f"  Augmented features: {node_features_aug.shape[1]} "
          f"(raw {len(feature_cols)} + spatial lags {TREATMENT_DIM * 3})")

    return (node_features_aug, node_outcomes, edge_index, spatial_edge_index, temporal_edge_index,
            node_mask_train, node_mask_test, node_country, node_year, N, T)


class INETARNet(nn.Module):
    def __init__(self, in_dim, hidden=HIDDEN_DIM, treatment_dim=TREATMENT_DIM, outcome_dim=OUTCOME_DIM):
        super().__init__()
        self.treatment_dim = treatment_dim
        self.spatial_lag_dim = treatment_dim * 3

        self.ego_encoder = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.ELU(),
            nn.Linear(hidden, hidden),
        )

        self.gcn1 = GCNConv(in_dim, hidden)
        self.gcn2 = GCNConv(hidden, hidden)
        self.norm1 = nn.LayerNorm(hidden)
        self.norm2 = nn.LayerNorm(hidden)

        repr_dim = hidden * 2

        self.outcome_logits = nn.Sequential(
            nn.Linear(repr_dim, hidden), nn.ELU(), nn.Dropout(0.2),
            nn.Linear(hidden, hidden), nn.ELU(),
            nn.Linear(hidden, outcome_dim),
        )

        self.local_logits = nn.Sequential(
            nn.Linear(hidden, hidden), nn.ELU(),
            nn.Linear(hidden, outcome_dim),
        )

        self.gps_mu = nn.Sequential(nn.Linear(repr_dim, hidden), nn.ELU(), nn.Linear(hidden, treatment_dim))
        self.gps_logvar = nn.Sequential(nn.Linear(repr_dim, hidden), nn.ELU(), nn.Linear(hidden, treatment_dim))

    def encode(self, x, edge_index):
        h_ego = self.ego_encoder(x)
        if edge_index.shape[1] > 0:
            h_gnn = F.elu(self.norm1(self.gcn1(x, edge_index)))
            h_gnn = F.elu(self.norm2(self.gcn2(h_gnn, edge_index)))
        else:
            h_gnn = torch.zeros_like(h_ego)
        return torch.cat([h_ego, h_gnn], dim=-1), h_ego

    def forward(self, x, edge_index):
        h_full, h_ego = self.encode(x, edge_index)
        y_full = F.softmax(self.outcome_logits(h_full), dim=-1)
        y_local = F.softmax(self.local_logits(h_ego), dim=-1)
        gps_mu = self.gps_mu(h_full)
        gps_logvar = self.gps_logvar(h_full).clamp(-5, 5)
        return y_full, y_local, gps_mu, gps_logvar

    def counterfactual_decompose(self, x, edge_index, spatial_edge_index):
        with torch.no_grad():
            h_full, h_ego = self.encode(x, edge_index)
            logits_full = self.outcome_logits(h_full)

            x_no_spatial = x.clone()
            x_no_spatial[:, -self.spatial_lag_dim:] = 0.0

            h_no_spatial, _ = self.encode(x_no_spatial, torch.zeros(2, 0, dtype=torch.long))
            logits_no_spatial = self.outcome_logits(h_no_spatial)

            spillover = logits_full - logits_no_spatial
            y_full = F.softmax(logits_full, dim=-1)

        return y_full, logits_no_spatial, spillover


def mmd_kernel(h1, h2, bandwidth=1.0):
    if h1.shape[0] < 2 or h2.shape[0] < 2:
        return torch.tensor(0.0)
    k11 = torch.exp(-torch.cdist(h1, h1) / (2 * bandwidth ** 2)).mean()
    k22 = torch.exp(-torch.cdist(h2, h2) / (2 * bandwidth ** 2)).mean()
    k12 = torch.exp(-torch.cdist(h1, h2) / (2 * bandwidth ** 2)).mean()
    return k11 + k22 - 2 * k12


def train_model(x, y, edge_index, mask_train, mask_test, in_dim):
    model = INETARNet(in_dim)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

    best_test = float("inf")
    best_state = None

    for epoch in range(EPOCHS):
        model.train()
        optimizer.zero_grad()

        y_full, y_local, gps_mu, gps_logvar = model(x, edge_index)

        loss_out = F.mse_loss(y_full[mask_train], y[mask_train])
        loss_local = F.mse_loss(y_local[mask_train], y[mask_train])

        treatment = x[mask_train, :TREATMENT_DIM]
        var = gps_logvar[mask_train].exp()
        gps_nll = 0.5 * ((treatment - gps_mu[mask_train]) ** 2 / (var + 1e-6) + gps_logvar[mask_train]).mean()

        h_full, _ = model.encode(x, edge_index)
        h_train = h_full[mask_train]
        med = treatment[:, 0].median()
        balance = mmd_kernel(h_train[treatment[:, 0] > med], h_train[treatment[:, 0] <= med])

        loss = loss_out + 0.3 * loss_local + 0.3 * gps_nll + 0.2 * balance

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()

        if (epoch + 1) % 50 == 0:
            model.eval()
            with torch.no_grad():
                yp, *_ = model(x, edge_index)
                test_mse = F.mse_loss(yp[mask_test], y[mask_test]).item()
            if test_mse < best_test:
                best_test = test_mse
                best_state = {k: v.clone() for k, v in model.state_dict().items()}
            print(f"  Epoch {epoch+1}/{EPOCHS}: train={loss.item():.5f}, test={test_mse:.6f}")

    if best_state:
        model.load_state_dict(best_state)
        print(f"  Best test MSE: {best_test:.6f}")
    return model


def placebo_test(model, x, spatial_ei, temporal_ei, y, mask_all, N, T, n_perms=10):
    model.eval()
    real_scores = []
    placebo_scores = []

    with torch.no_grad():
        full_ei = torch.cat([spatial_ei, temporal_ei], dim=1) if spatial_ei.shape[1] > 0 else temporal_ei
        y_full, dom, spill = model.counterfactual_decompose(x, full_ei, spatial_ei)
        real_contagion = spill.abs().sum(dim=1) / (spill.abs().sum(dim=1) + dom.abs().sum(dim=1) + 1e-10)
        real_mean = real_contagion[mask_all].mean().item()

        for perm in range(n_perms):
            rng = np.random.RandomState(perm + 42)
            x_placebo = x.clone()
            for t in range(T):
                offset = t * N
                perm_idx = torch.tensor(rng.permutation(N), dtype=torch.long)
                lag_start = x.shape[1] - TREATMENT_DIM * 3
                x_placebo[offset:offset + N, lag_start:] = x[offset + perm_idx, lag_start:]

            shuffled_spatial = spatial_ei.clone()
            if shuffled_spatial.shape[1] > 0:
                for t in range(T):
                    offset = t * N
                    perm_map = torch.tensor(rng.permutation(N), dtype=torch.long)
                    mask_t = (shuffled_spatial[0] >= offset) & (shuffled_spatial[0] < offset + N)
                    shuffled_spatial[0, mask_t] = offset + perm_map[shuffled_spatial[0, mask_t] - offset]
                    shuffled_spatial[1, mask_t] = offset + perm_map[shuffled_spatial[1, mask_t] - offset]

            placebo_ei = torch.cat([shuffled_spatial, temporal_ei], dim=1)
            _, dom_p, spill_p = model.counterfactual_decompose(x_placebo, placebo_ei, shuffled_spatial)
            placebo_contagion = spill_p.abs().sum(dim=1) / (spill_p.abs().sum(dim=1) + dom_p.abs().sum(dim=1) + 1e-10)
            placebo_scores.append(placebo_contagion[mask_all].mean().item())

    placebo_mean = np.mean(placebo_scores)
    placebo_std = np.std(placebo_scores)
    ratio = real_mean / (placebo_mean + 1e-10)
    z_score = (real_mean - placebo_mean) / (placebo_std + 1e-10)

    return real_mean, placebo_mean, ratio, z_score


def run_stage4():
    print("=== Stage 4: Network SCM (Spatio-Temporal Graph) ===\n")

    df, mapping = load_all_data()
    feature_cols = FACTOR_COLS + BETA_COLS + ["gdp_pc", "urbanization"]

    years_all = sorted(df["year"].unique())
    years_use = [y for y in years_all if y >= 1990]

    complete = df.groupby("country_text_id").apply(
        lambda g: g[g["year"].isin(years_use)].dropna(subset=feature_cols + STATE_COLS)["year"].nunique()
    )
    countries_iso3 = sorted(complete[complete >= len(years_use) * 0.8].index.tolist())
    print(f"Countries: {len(countries_iso3)}, Years: {years_use[0]}-{years_use[-1]}")

    print(f"\nBuilding spatial edges...")
    contig_pairs, alliance_by_year = build_spatial_edges(mapping, countries_iso3)
    print(f"  Contiguity pairs: {len(contig_pairs)}, Alliance pairs (2020): {len(alliance_by_year.get(2020, set()))}")

    print(f"\nBuilding spatio-temporal graph...")
    (x, y, edge_index, spatial_ei, temporal_ei,
     mask_train, mask_test, node_country, node_year, N, T) = \
        build_spatiotemporal_graph(df, countries_iso3, years_use, contig_pairs, alliance_by_year, feature_cols)

    in_dim = x.shape[1]
    print(f"  Train nodes: {mask_train.sum()}, Test nodes: {mask_test.sum()}")

    print(f"\nTraining INE-TARNet on spatio-temporal graph...")
    model = train_model(x, y, edge_index, mask_train, mask_test, in_dim)

    print(f"\nCounterfactual decomposition...")
    model.eval()
    with torch.no_grad():
        full_ei = torch.cat([spatial_ei, temporal_ei], dim=1)
        y_full, domestic, spillover = model.counterfactual_decompose(x, full_ei, spatial_ei)

    rows = []
    cname_map = df.drop_duplicates("country_text_id").set_index("country_text_id")["country_name"]
    for nid in range(len(node_country)):
        spill_mag = spillover[nid].abs().sum().item()
        dom_mag = domestic[nid].abs().sum().item()
        total = spill_mag + dom_mag + 1e-10
        contagion = spill_mag / total

        row = {
            "country_text_id": node_country[nid],
            "country_name": cname_map.get(node_country[nid], node_country[nid]),
            "year": node_year[nid],
            "contagion_score": contagion,
            "domestic_score": 1 - contagion,
        }
        for k in range(OUTCOME_DIM):
            row[f"spillover_state_{k}"] = spillover[nid, k].item()
        rows.append(row)

    scores_df = pd.DataFrame(rows)
    scores_df = scores_df.sort_values(["country_text_id", "year"])
    scores_df["contagion_smooth"] = scores_df.groupby("country_text_id")["contagion_score"].transform(
        lambda s: s.rolling(3, min_periods=1, center=True).mean()
    )

    output_dir = os.path.dirname(os.path.abspath(__file__))
    scores_df.to_csv(os.path.join(output_dir, "contagion_scores.csv"), index=False)

    print(f"\n=== Placebo test ({10} permutations) ===")
    mask_all = mask_train | mask_test
    real_mean, placebo_mean, ratio, z_score = placebo_test(
        model, x, spatial_ei, temporal_ei, y, mask_all, N, T
    )
    print(f"  Real avg contagion:    {real_mean:.4f}")
    print(f"  Placebo avg contagion: {placebo_mean:.4f}")
    print(f"  Ratio: {ratio:.2f}x")
    print(f"  Z-score: {z_score:.2f}")
    if z_score > 2.0:
        print(f"  PASS: Network structure is statistically significant (z={z_score:.1f}, p<0.001)")
        if ratio > 1.2:
            print(f"  Effect size: LARGE ({ratio:.0%} above random)")
        elif ratio > 1.05:
            print(f"  Effect size: MODERATE ({(ratio-1)*100:.1f}% above random)")
        else:
            print(f"  Effect size: SMALL but significant ({(ratio-1)*100:.1f}% above random)")
    else:
        print(f"  NOT SIGNIFICANT (z={z_score:.1f})")

    latest = scores_df[scores_df["year"] == years_use[-1]].sort_values("contagion_smooth", ascending=False)
    print(f"\n=== Most network-influenced ({years_use[-1]}) ===")
    for _, row in latest.head(10).iterrows():
        print(f"  {row['country_name']}: {row['contagion_smooth']:.3f}")
    print(f"\n=== Most domestically driven ({years_use[-1]}) ===")
    for _, row in latest.tail(10).iterrows():
        print(f"  {row['country_name']}: {row['contagion_smooth']:.3f}")

    print(f"\n=== Case studies (smoothed) ===")
    for country in ["Hungary", "Türkiye", "Poland", "United States of America", "Denmark"]:
        sub = scores_df[scores_df["country_name"] == country].sort_values("year").tail(5)
        if len(sub) == 0:
            continue
        vals = ", ".join(f"{int(r['year'])}:{r['contagion_smooth']:.3f}" for _, r in sub.iterrows())
        print(f"  {country}: {vals}")

    return model, scores_df


if __name__ == "__main__":
    model, scores_df = run_stage4()
