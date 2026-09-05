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
TRAIN_CUTOFF = int(os.environ.get("AIM4D_CUTOFF", "2019"))
EXCLUDE_COUNTRY = os.environ.get("AIM4D_EXCLUDE_COUNTRY", "").strip() or None


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
        train_med = macro_sub.loc[macro_sub["year"] <= TRAIN_CUTOFF, c].median()
        if pd.isna(train_med):
            train_med = 0.0
        macro_sub[c] = macro_sub[c].fillna(train_med)
    df = df.merge(macro_sub, left_on=["country_text_id", "year"], right_on=["iso3", "year"], how="left")
    for c in macro_cols:
        train_med = df.loc[df["year"] <= TRAIN_CUTOFF, c].median()
        if pd.isna(train_med):
            train_med = 0.0
        df[c] = df[c].fillna(train_med)
    return df, mapping


def build_spatial_edges(mapping, countries_iso3):
    cow_map = dict(zip(mapping["country_text_id"], mapping["COWcode"]))
    iso3_map = {v: k for k, v in cow_map.items()}

    base = os.path.dirname(os.path.abspath(__file__))
    cont = pd.read_csv(os.path.join(base, "..", "data", "contiguity", "DirectContiguity320", "contdird.csv"), low_memory=False)
    land = cont[cont["conttype"] <= 2].groupby(["state1no", "state2no"]).last().reset_index()
    contig_pairs = set()
    for _, row in land.iterrows():
        s1, s2 = iso3_map.get(row["state1no"]), iso3_map.get(row["state2no"])
        if s1 in countries_iso3 and s2 in countries_iso3:
            contig_pairs.add((countries_iso3.index(s1), countries_iso3.index(s2)))
            contig_pairs.add((countries_iso3.index(s2), countries_iso3.index(s1)))

    base_atop = os.path.dirname(os.path.abspath(__file__))
    atop = pd.read_csv(os.path.join(base_atop, "..", "data", "atop", "ATOP 5.1 (.csv)", "atop5_1dy.csv"), low_memory=False, encoding="latin-1")
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

    cultural_pairs = set()
    cult_path = os.path.join(base_atop, "..", "data", "cultural_pairs.csv")
    if os.path.exists(cult_path):
        cult = pd.read_csv(cult_path)
        for _, row in cult.iterrows():
            a, b = row["iso3_a"], row["iso3_b"]
            if a in countries_iso3 and b in countries_iso3:
                cultural_pairs.add((countries_iso3.index(a), countries_iso3.index(b)))
                cultural_pairs.add((countries_iso3.index(b), countries_iso3.index(a)))

    return contig_pairs, alliance_by_year, cultural_pairs


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


def build_spatiotemporal_graph(df, countries_iso3, years, contig_pairs, alliance_by_year,
                                feature_cols, cultural_pairs=None):
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
    spatial_lag_cultural = torch.zeros(total_nodes, TREATMENT_DIM)

    contig_edges_src, contig_edges_dst = [], []
    alliance_edges_src, alliance_edges_dst = [], []
    trade_edges_src, trade_edges_dst = [], []
    cultural_edges_src, cultural_edges_dst = [], []
    temporal_edges_src, temporal_edges_dst = [], []

    if cultural_pairs is None:
        cultural_pairs = set()

    for t, year in enumerate(years):
        offset = t * N

        for (i, j) in contig_pairs:
            contig_edges_src.append(offset + i)
            contig_edges_dst.append(offset + j)

        ally_pairs = alliance_by_year.get(year, set())
        for (i, j) in ally_pairs:
            alliance_edges_src.append(offset + i)
            alliance_edges_dst.append(offset + j)

        for (i, j) in cultural_pairs:
            cultural_edges_src.append(offset + i)
            cultural_edges_dst.append(offset + j)

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

        if t > 0:
            prev_offset = (t - 1) * N
            treat_lag = treatment[prev_offset:prev_offset + N]
        else:
            treat_lag = treatment[offset:offset + N]

        spatial_lag_contig[offset:offset + N] = neighbor_mean(
            treat_lag, [i for i, j in contig_pairs], [j for i, j in contig_pairs], N
        )
        ally_src = [i for i, j in ally_pairs]
        ally_dst = [j for i, j in ally_pairs]
        spatial_lag_alliance[offset:offset + N] = neighbor_mean(treat_lag, ally_src, ally_dst, N)

        t_src = [s - offset for s in trade_edges_src if offset <= s < offset + N]
        t_dst = [d - offset for d in trade_edges_dst if offset <= d < offset + N]
        spatial_lag_trade[offset:offset + N] = neighbor_mean(treat_lag, t_src, t_dst, N)

        cult_src = [i for i, j in cultural_pairs]
        cult_dst = [j for i, j in cultural_pairs]
        spatial_lag_cultural[offset:offset + N] = neighbor_mean(treat_lag, cult_src, cult_dst, N)

    node_features_aug = torch.cat([
        node_features,
        spatial_lag_contig,
        spatial_lag_alliance,
        spatial_lag_trade,
        spatial_lag_cultural,
    ], dim=-1)

    all_spatial_src = (contig_edges_src + alliance_edges_src
                       + trade_edges_src + cultural_edges_src)
    all_spatial_dst = (contig_edges_dst + alliance_edges_dst
                       + trade_edges_dst + cultural_edges_dst)
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
          f"(raw {len(feature_cols)} + spatial lags {TREATMENT_DIM * 4})  "
          f"[contig + alliance + trade + cultural]")

    return (node_features_aug, node_outcomes, edge_index, spatial_edge_index, temporal_edge_index,
            node_mask_train, node_mask_test, node_country, node_year, N, T)


class HeterogeneousExposure(nn.Module):
    def __init__(self, hidden):
        super().__init__()
        self.weight_net = nn.Sequential(
            nn.Linear(hidden * 2 + 1, 16), nn.ELU(),
            nn.Linear(16, 1), nn.Sigmoid(),
        )

    def forward(self, h, edge_index):
        if edge_index.shape[1] == 0:
            return torch.zeros(h.shape[0], h.shape[1], device=h.device)
        src, dst = edge_index
        n = h.shape[0]
        sim = torch.exp(-((h[dst] - h[src]) ** 2).sum(dim=-1, keepdim=True) / (2 * h.shape[1]))
        pair_feat = torch.cat([h[dst], h[src], sim], dim=-1)
        w = self.weight_net(pair_feat).squeeze(-1)
        weighted = h[src] * w.unsqueeze(-1)
        agg = torch.zeros(n, h.shape[1], device=h.device)
        agg.scatter_add_(0, dst.unsqueeze(-1).expand_as(weighted), weighted)
        w_sum = torch.zeros(n, device=h.device).scatter_add_(0, dst, w).clamp(min=1e-8)
        return agg / w_sum.unsqueeze(-1)


class INETARNet(nn.Module):
    def __init__(self, in_dim, hidden=HIDDEN_DIM, treatment_dim=TREATMENT_DIM, outcome_dim=OUTCOME_DIM,
                 n_edge_types=4):
        super().__init__()
        self.treatment_dim = treatment_dim
        self.n_edge_types = n_edge_types
        self.spatial_lag_dim = treatment_dim * n_edge_types

        self.w_logits = nn.Parameter(torch.zeros(n_edge_types))

        self.ego_encoder = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.ELU(),
            nn.Linear(hidden, hidden),
        )

        self.gcn1 = GCNConv(in_dim, hidden)
        self.gcn2 = GCNConv(hidden, hidden)
        self.norm1 = nn.LayerNorm(hidden)
        self.norm2 = nn.LayerNorm(hidden)
        self.exposure = HeterogeneousExposure(hidden)

        repr_dim = hidden * 3

        self.outcome_logits = nn.Sequential(
            nn.Linear(repr_dim, hidden), nn.ELU(), nn.Dropout(0.2),
            nn.Linear(hidden, hidden), nn.ELU(),
            nn.Linear(hidden, outcome_dim),
        )

        self.local_logits = nn.Sequential(
            nn.Linear(hidden, hidden), nn.ELU(),
            nn.Linear(hidden, outcome_dim),
        )

        # A comparator that never sees neighbour-derived features. The ego head
        # above bypasses message passing but still receives the weighted spatial
        # lags, so it cannot stand for "no network information". This one is fed
        # the own-country block alone.
        self.domestic_encoder = nn.Sequential(
            nn.Linear(in_dim - self.spatial_lag_dim, hidden), nn.ELU(),
            nn.Linear(hidden, hidden),
        )
        self.domestic_logits = nn.Sequential(
            nn.Linear(hidden, hidden), nn.ELU(),
            nn.Linear(hidden, outcome_dim),
        )

        self.gps_mu = nn.Sequential(nn.Linear(repr_dim, hidden), nn.ELU(), nn.Linear(hidden, treatment_dim))
        self.gps_logvar = nn.Sequential(nn.Linear(repr_dim, hidden), nn.ELU(), nn.Linear(hidden, treatment_dim))

    def get_w_weights(self):
        """Return learned convex combination weights for the three W matrices."""
        return F.softmax(self.w_logits, dim=0)

    def apply_learned_w(self, x):
        """Reweight the spatial-lag blocks using learned α weights.

        Generalized to n_edge_types: each lag block contributes alpha[k] * lag_k
        to the weighted composite, which is then replicated n_edge_types times
        to preserve input dimensionality (the GCN downstream expects the same
        feature width as before).
        """
        alpha = self.get_w_weights()
        base_dim = x.shape[1] - self.spatial_lag_dim
        x_base = x[:, :base_dim]
        n_e = self.n_edge_types
        weighted_lag = torch.zeros(x.shape[0], self.treatment_dim, device=x.device)
        for k in range(n_e):
            block = x[:, base_dim + k * self.treatment_dim:base_dim + (k + 1) * self.treatment_dim]
            weighted_lag = weighted_lag + alpha[k] * block
        return torch.cat([x_base] + [weighted_lag] * n_e, dim=-1)

    def encode(self, x, edge_index):
        x_weighted = self.apply_learned_w(x)
        h_ego = self.ego_encoder(x_weighted)
        if edge_index.shape[1] > 0:
            h_gnn = F.elu(self.norm1(self.gcn1(x_weighted, edge_index)))
            h_gnn = F.elu(self.norm2(self.gcn2(h_gnn, edge_index)))
            h_exp = self.exposure(h_gnn, edge_index)
        else:
            h_gnn = torch.zeros_like(h_ego)
            h_exp = torch.zeros_like(h_ego)
        return torch.cat([h_ego, h_gnn, h_exp], dim=-1), h_ego

    def domestic_only(self, x):
        base_dim = x.shape[1] - self.spatial_lag_dim
        return F.softmax(self.domestic_logits(self.domestic_encoder(x[:, :base_dim])), dim=-1)

    def forward(self, x, edge_index):
        h_full, h_ego = self.encode(x, edge_index)
        y_full = F.softmax(self.outcome_logits(h_full), dim=-1)
        y_local = F.softmax(self.local_logits(h_ego), dim=-1)
        y_domestic = self.domestic_only(x)
        gps_mu = self.gps_mu(h_full)
        gps_logvar = self.gps_logvar(h_full).clamp(-5, 5)
        return y_full, y_local, y_domestic, gps_mu, gps_logvar

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


def split_val_test(mask_test, node_year):
    """Split the post-cutoff block into a selection set and an untouched test set.

    Retaining the parameter state that minimises error on the evaluation rows
    makes those rows a selection set, so the resulting figure is not an
    out-of-sample estimate. The earlier post-cutoff years select; the later ones
    are scored once and never used for selection.
    """
    yrs = torch.as_tensor(node_year)[mask_test]
    if yrs.numel() == 0:
        return mask_test, mask_test
    cut = int(torch.quantile(yrs.float(), 0.5).item())
    mv = mask_test & (torch.as_tensor(node_year) <= cut)
    mt = mask_test & (torch.as_tensor(node_year) > cut)
    if mt.sum() == 0 or mv.sum() == 0:
        return mask_test, mask_test
    return mv, mt


def train_model(x, y, edge_index, mask_train, mask_test, in_dim, seed=42, mask_val=None,
                n_edge_types=4):
    torch.manual_seed(seed)
    np.random.seed(seed)
    model = INETARNet(in_dim, n_edge_types=n_edge_types)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

    best_test = float("inf")
    best_state = None

    for epoch in range(EPOCHS):
        model.train()
        optimizer.zero_grad()

        y_full, y_local, y_domestic, gps_mu, gps_logvar = model(x, edge_index)

        loss_out = F.mse_loss(y_full[mask_train], y[mask_train])
        loss_local = F.mse_loss(y_local[mask_train], y[mask_train])
        loss_dom = F.mse_loss(y_domestic[mask_train], y[mask_train])

        treatment = x[mask_train, :TREATMENT_DIM]
        var = gps_logvar[mask_train].exp()
        gps_nll = 0.5 * ((treatment - gps_mu[mask_train]) ** 2 / (var + 1e-6) + gps_logvar[mask_train]).mean()

        h_full, _ = model.encode(x, edge_index)
        h_train = h_full[mask_train]
        med = treatment[:, 0].median()
        balance = mmd_kernel(h_train[treatment[:, 0] > med], h_train[treatment[:, 0] <= med])

        loss = loss_out + 0.3 * loss_local + 0.3 * loss_dom + 0.3 * gps_nll + 0.2 * balance

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()

        if (epoch + 1) % 50 == 0:
            model.eval()
            with torch.no_grad():
                yp, *_ = model(x, edge_index)
                sel = mask_val if mask_val is not None else mask_test
                test_mse = F.mse_loss(yp[sel], y[sel]).item()
            if test_mse < best_test:
                best_test = test_mse
                best_state = {k: v.clone() for k, v in model.state_dict().items()}
            print(f"  Epoch {epoch+1}/{EPOCHS}: train={loss.item():.5f}, test={test_mse:.6f}")

    if best_state:
        model.load_state_dict(best_state)
        print(f"  Best test MSE: {best_test:.6f}")
    return model


def network_ablation_test(model, x, y, edge_index, spatial_ei, temporal_ei,
                          mask_train, mask_test, in_dim):
    model.eval()
    with torch.no_grad():
        full_ei = torch.cat([spatial_ei, temporal_ei], dim=1)
        y_full, _, _, _, _ = model(x, full_ei)
        mse_full = F.mse_loss(y_full[mask_test], y[mask_test]).item()

        y_temporal, _, _, _, _ = model(x, temporal_ei)
        mse_temporal = F.mse_loss(y_temporal[mask_test], y[mask_test]).item()

        x_no_lag = x.clone()
        x_no_lag[:, -model.spatial_lag_dim:] = 0.0
        y_no_lag, _, _, _, _ = model(x_no_lag, temporal_ei)
        mse_no_network = F.mse_loss(y_no_lag[mask_test], y[mask_test]).item()

    improvement_spatial = (mse_temporal - mse_full) / mse_temporal * 100
    improvement_total = (mse_no_network - mse_full) / mse_no_network * 100

    return {
        "mse_full": mse_full,
        "mse_temporal_only": mse_temporal,
        "mse_no_network": mse_no_network,
        "improvement_spatial_edges": improvement_spatial,
        "improvement_total_network": improvement_total,
    }


def run_stage4(seed=42, write_outputs=True):
    print(f"=== Stage 4: Network SCM (Spatio-Temporal Graph) — seed={seed} ===\n")

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
    contig_pairs, alliance_by_year, cultural_pairs = build_spatial_edges(mapping, countries_iso3)
    print(f"  Contiguity pairs: {len(contig_pairs)}, "
          f"Alliance pairs (2020): {len(alliance_by_year.get(2020, set()))}, "
          f"Cultural pairs: {len(cultural_pairs)}")

    print(f"\nBuilding spatio-temporal graph...")
    (x, y, edge_index, spatial_ei, temporal_ei,
     mask_train, mask_test, node_country, node_year, N, T) = \
        build_spatiotemporal_graph(df, countries_iso3, years_use, contig_pairs, alliance_by_year,
                                    feature_cols, cultural_pairs=cultural_pairs)

    in_dim = x.shape[1]
    print(f"  Train nodes: {mask_train.sum()}, Test nodes: {mask_test.sum()}")

    print(f"\nTraining INE-TARNet on spatio-temporal graph...")
    mask_val, mask_eval = split_val_test(mask_test, node_year)
    print(f"  selection rows {int(mask_val.sum())}, untouched test rows {int(mask_eval.sum())}")
    model = train_model(x, y, edge_index, mask_train, mask_eval, in_dim, seed=seed, mask_val=mask_val)
    with torch.no_grad():
        yp, *_ = model(x, torch.cat([spatial_ei, temporal_ei], dim=1))
        print(f"  MSE on untouched test rows: {F.mse_loss(yp[mask_eval], y[mask_eval]).item():.6f}")

    w_weights = model.get_w_weights().detach().numpy()
    w_names = ["contiguity", "alliance", "trade", "cultural"][:len(w_weights)]
    print(f"\n  Learned convex W weights (Neumayer & Plumper 2016):")
    for name, w in zip(w_names, w_weights):
        print(f"    α_{name}: {w:.3f}")

    output_dir = os.path.dirname(os.path.abspath(__file__))

    if not write_outputs:
        return model, None

    print(f"\nCounterfactual decomposition...")
    model.eval()
    with torch.no_grad():
        full_ei = torch.cat([spatial_ei, temporal_ei], dim=1)
        y_full, domestic, spillover = model.counterfactual_decompose(x, full_ei, spatial_ei)

        y_pred_full, y_pred_local, y_pred_dom, _, _ = model(x, full_ei)
        nscm_resid_full = (y - y_pred_full).numpy()
        nscm_resid_domestic = (y - y_pred_local).numpy()
        nscm_resid_truedom = (y - y_pred_dom).numpy()

    resid_rows = []
    for nid in range(len(node_country)):
        rrow = {"country_text_id": node_country[nid], "year": node_year[nid]}
        for k in range(OUTCOME_DIM):
            rrow[f"nscm_resid_full_{k}"] = nscm_resid_full[nid, k]
            rrow[f"nscm_resid_domestic_{k}"] = nscm_resid_domestic[nid, k]
            rrow[f"nscm_resid_truedom_{k}"] = nscm_resid_truedom[nid, k]
        resid_rows.append(rrow)
    resid_df = pd.DataFrame(resid_rows)
    resid_df.to_csv(os.path.join(output_dir, "nscm_residuals.csv"), index=False)
    print(f"Saved NSCM residuals ({len(resid_df)} rows)")

    print(f"Computing per-factor contagion scores...")
    with torch.no_grad():
        h_full, _ = model.encode(x, full_ei)
        logits_full = model.outcome_logits(h_full)

        factor_spillovers = {}
        for fk in range(TREATMENT_DIM):
            x_partial = x.clone()
            lag_start = x.shape[1] - model.spatial_lag_dim
            for edge_type in range(model.n_edge_types):
                col_idx = lag_start + edge_type * TREATMENT_DIM + fk
                if col_idx < x_partial.shape[1]:
                    x_partial[:, col_idx] = 0.0
            h_partial, _ = model.encode(x_partial, full_ei)
            logits_partial = model.outcome_logits(h_partial)
            factor_spillovers[fk] = logits_full - logits_partial

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
        for fk in range(TREATMENT_DIM):
            fk_mag = factor_spillovers[fk][nid].abs().sum().item()
            row[f"contagion_factor_{fk+1}"] = fk_mag / (total + 1e-10)
        rows.append(row)

    scores_df = pd.DataFrame(rows)
    scores_df = scores_df.sort_values(["country_text_id", "year"])
    scores_df["contagion_smooth"] = scores_df.groupby("country_text_id")["contagion_score"].transform(
        lambda s: s.rolling(3, min_periods=1).mean()
    )

    output_dir = os.path.dirname(os.path.abspath(__file__))
    scores_df.to_csv(os.path.join(output_dir, "contagion_scores.csv"), index=False)

    print(f"\n=== Network ablation test ===")
    ablation = network_ablation_test(model, x, y, edge_index, spatial_ei, temporal_ei,
                                     mask_train, mask_test, in_dim)
    print(f"  MSE (full model):       {ablation['mse_full']:.6f}")
    print(f"  MSE (temporal only):    {ablation['mse_temporal_only']:.6f}")
    print(f"  MSE (no network):       {ablation['mse_no_network']:.6f}")
    print(f"  Improvement from spatial edges: {ablation['improvement_spatial_edges']:.1f}%")
    print(f"  Improvement from total network: {ablation['improvement_total_network']:.1f}%")
    if ablation["improvement_total_network"] > 5:
        print(f"  PASS: Network features improve prediction by {ablation['improvement_total_network']:.1f}%")
    elif ablation["improvement_total_network"] > 1:
        print(f"  MODERATE: Network contributes {ablation['improvement_total_network']:.1f}% improvement")
    else:
        print(f"  WEAK: Network adds <1% improvement")

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
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()
    model, scores_df = run_stage4(seed=args.seed)
