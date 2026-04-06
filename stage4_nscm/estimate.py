import sys
import os
import warnings
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

FACTOR_COLS = ["factor_1", "factor_2", "factor_3", "factor_4"]
BETA_COLS = ["beta_factor_1", "beta_factor_2", "beta_factor_3", "beta_factor_4"]
STATE_COLS = ["prob_state_0", "prob_state_1", "prob_state_2", "prob_state_3", "prob_state_4"]
TREATMENT_DIM = 4
OUTCOME_DIM = 5
HIDDEN_DIM = 32
EPOCHS = 150
LR = 5e-3


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


def build_contiguity_edges(mapping, countries_iso3):
    cow_map = dict(zip(mapping["country_text_id"], mapping["COWcode"]))
    iso3_map = {v: k for k, v in cow_map.items()}

    cont = pd.read_csv("data/contiguity/DirectContiguity320/contdird.csv", low_memory=False)
    land = cont[cont["conttype"] <= 2]

    latest = land.groupby(["state1no", "state2no"]).last().reset_index()

    edges = []
    for _, row in latest.iterrows():
        s1 = iso3_map.get(row["state1no"])
        s2 = iso3_map.get(row["state2no"])
        if s1 in countries_iso3 and s2 in countries_iso3:
            i = countries_iso3.index(s1)
            j = countries_iso3.index(s2)
            edges.append((i, j))
            edges.append((j, i))

    if edges:
        edge_index = torch.tensor(edges, dtype=torch.long).t()
    else:
        edge_index = torch.zeros(2, 0, dtype=torch.long)
    return edge_index


def build_alliance_edges(mapping, countries_iso3, year):
    cow_map = dict(zip(mapping["country_text_id"], mapping["COWcode"]))
    iso3_map = {v: k for k, v in cow_map.items()}

    atop = pd.read_csv("data/atop/ATOP 5.1 (.csv)/atop5_1dy.csv", low_memory=False, encoding="latin-1")
    active = atop[(atop["atopally"] == 1) & (atop["year"] >= year - 5) & (atop["year"] <= year)]

    edges = set()
    for _, row in active.iterrows():
        s1 = iso3_map.get(row["mem1"])
        s2 = iso3_map.get(row["mem2"])
        if s1 in countries_iso3 and s2 in countries_iso3:
            i = countries_iso3.index(s1)
            j = countries_iso3.index(s2)
            edges.add((i, j))
            edges.add((j, i))

    if edges:
        edge_index = torch.tensor(list(edges), dtype=torch.long).t()
    else:
        edge_index = torch.zeros(2, 0, dtype=torch.long)
    return edge_index


def build_trade_edges(df_year, countries_iso3, k=5):
    trade_vals = df_year.set_index("country_text_id")["gdp_pc"].reindex(countries_iso3).fillna(0).values
    n = len(countries_iso3)

    if n < k + 1:
        return torch.zeros(2, 0, dtype=torch.long)

    log_vals = np.log1p(np.abs(trade_vals))
    diffs = np.abs(log_vals[:, None] - log_vals[None, :])
    np.fill_diagonal(diffs, np.inf)

    edges = []
    for i in range(n):
        neighbors = np.argsort(diffs[i])[:k]
        for j in neighbors:
            edges.append((i, j))
            edges.append((j, i))

    edges = list(set(edges))
    return torch.tensor(edges, dtype=torch.long).t()


def compute_network_exposure(treatment, edge_index):
    if edge_index.shape[1] == 0:
        return torch.zeros_like(treatment)
    src, dst = edge_index
    n = treatment.shape[0]
    exposure = torch.zeros_like(treatment)
    deg = torch.zeros(n, device=treatment.device)
    exposure.scatter_add_(0, dst.unsqueeze(-1).expand_as(treatment[src]), treatment[src])
    deg.scatter_add_(0, dst, torch.ones(src.shape[0], device=treatment.device))
    deg = deg.clamp(min=1)
    return exposure / deg.unsqueeze(-1)


class LocalEncoder(nn.Module):
    def __init__(self, in_dim, hidden_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ELU(),
        )

    def forward(self, x):
        return self.net(x)


class GraphEncoder(nn.Module):
    def __init__(self, in_dim, hidden_dim):
        super().__init__()
        self.conv_contig = GCNConv(in_dim, hidden_dim)
        self.conv_alliance = GCNConv(in_dim, hidden_dim)
        self.conv_trade = GCNConv(in_dim, hidden_dim)
        self.norm = nn.LayerNorm(hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)

    def forward(self, x, contig_ei, alliance_ei, trade_ei):
        h1 = self.conv_contig(x, contig_ei) if contig_ei.shape[1] > 0 else torch.zeros(x.shape[0], self.conv_contig.out_channels, device=x.device)
        h2 = self.conv_alliance(x, alliance_ei) if alliance_ei.shape[1] > 0 else torch.zeros_like(h1)
        h3 = self.conv_trade(x, trade_ei) if trade_ei.shape[1] > 0 else torch.zeros_like(h1)
        h = F.elu(self.norm(h1 + h2 + h3))
        h = F.elu(self.norm2(self.conv2(h, contig_ei) if contig_ei.shape[1] > 0 else h))
        return h


class OutcomeHead(nn.Module):
    def __init__(self, repr_dim, treatment_dim, exposure_dim, outcome_dim, hidden=32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(repr_dim + treatment_dim + exposure_dim, hidden),
            nn.ELU(),
            nn.Dropout(0.2),
            nn.Linear(hidden, outcome_dim),
            nn.Softmax(dim=-1),
        )

    def forward(self, h, treatment, exposure):
        return self.net(torch.cat([h, treatment, exposure], dim=-1))


class ContagionNet(nn.Module):
    def __init__(self, in_dim, hidden_dim=HIDDEN_DIM, treatment_dim=TREATMENT_DIM,
                 exposure_dim=TREATMENT_DIM * 3, outcome_dim=OUTCOME_DIM):
        super().__init__()
        self.treatment_dim = treatment_dim

        self.graph_encoder = GraphEncoder(in_dim, hidden_dim)
        self.local_encoder = LocalEncoder(in_dim, hidden_dim)

        self.full_head = OutcomeHead(hidden_dim, treatment_dim, exposure_dim, outcome_dim)
        self.local_head = OutcomeHead(hidden_dim, treatment_dim, 0, outcome_dim)

    def forward(self, x, contig_ei, alliance_ei, trade_ei):
        treatment = x[:, :self.treatment_dim]

        h_graph = self.graph_encoder(x, contig_ei, alliance_ei, trade_ei)

        exp_contig = compute_network_exposure(treatment, contig_ei)
        exp_alliance = compute_network_exposure(treatment, alliance_ei)
        exp_trade = compute_network_exposure(treatment, trade_ei)
        exposure = torch.cat([exp_contig, exp_alliance, exp_trade], dim=-1)

        y_full = self.full_head(h_graph, treatment, exposure)

        h_local = self.local_encoder(x)
        y_local = self.local_head(h_local, treatment, torch.zeros(0))

        return y_full, y_local, exposure

    def predict_contagion(self, x, contig_ei, alliance_ei, trade_ei):
        with torch.no_grad():
            y_full, y_local, exposure = self.forward(x, contig_ei, alliance_ei, trade_ei)
            spillover = y_full - y_local
        return y_full, y_local, spillover, exposure


def build_snapshot(df_year, countries_iso3, mapping, feature_cols, year):
    x_data = df_year.set_index("country_text_id").reindex(countries_iso3)

    x = torch.tensor(x_data[feature_cols].values, dtype=torch.float32)
    y = torch.tensor(x_data[STATE_COLS].values, dtype=torch.float32)

    contig_ei = build_contiguity_edges(mapping, countries_iso3)
    alliance_ei = build_alliance_edges(mapping, countries_iso3, year)
    trade_ei = build_trade_edges(df_year, countries_iso3)

    return x, y, contig_ei, alliance_ei, trade_ei


def train_model(snapshots, in_dim, epochs=EPOCHS):
    model = ContagionNet(in_dim)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    for epoch in range(epochs):
        model.train()
        total_loss = 0.0

        for x, y, c_ei, a_ei, t_ei in snapshots:
            optimizer.zero_grad()
            y_full, y_local, exposure = model(x, c_ei, a_ei, t_ei)

            loss_full = F.mse_loss(y_full, y)
            loss_local = F.mse_loss(y_local, y)
            loss = loss_full + 0.5 * loss_local

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += loss.item()

        scheduler.step()

        if (epoch + 1) % 30 == 0:
            avg = total_loss / len(snapshots)
            print(f"  Epoch {epoch+1}/{epochs}: loss={avg:.6f}")

    return model


def compute_contagion_scores(model, snapshots, df, countries_iso3, years):
    model.eval()
    rows = []

    for t, (x, y, c_ei, a_ei, t_ei) in enumerate(snapshots):
        y_full, y_local, spillover, exposure = model.predict_contagion(x, c_ei, a_ei, t_ei)

        for i, iso3 in enumerate(countries_iso3):
            contagion_magnitude = spillover[i].abs().sum().item()
            domestic_magnitude = y_local[i].abs().sum().item()
            total = contagion_magnitude + domestic_magnitude + 1e-10
            contagion_ratio = contagion_magnitude / total

            row = {
                "country_text_id": iso3,
                "year": int(years[t]),
                "contagion_score": contagion_ratio,
                "domestic_score": 1 - contagion_ratio,
                "contagion_magnitude": contagion_magnitude,
            }
            for k in range(TREATMENT_DIM):
                row[f"spillover_factor_{k+1}"] = spillover[i, k].item() if k < spillover.shape[1] else 0.0
            for k in range(TREATMENT_DIM):
                row[f"exposure_contig_f{k+1}"] = exposure[i, k].item()
                row[f"exposure_alliance_f{k+1}"] = exposure[i, TREATMENT_DIM + k].item()
                row[f"exposure_trade_f{k+1}"] = exposure[i, 2 * TREATMENT_DIM + k].item()

            rows.append(row)

    return pd.DataFrame(rows)


def run_stage4():
    print("=== Stage 4: Network Structural Causal Model ===\n")

    df, mapping = load_all_data()
    feature_cols = FACTOR_COLS + BETA_COLS + ["gdp_pc", "urbanization"]
    in_dim = len(feature_cols)

    countries = sorted(df["country_text_id"].unique())
    years_all = sorted(df["year"].unique())
    years_use = [y for y in years_all if y >= 1990]

    complete = df.groupby("country_text_id").apply(
        lambda g: g[g["year"].isin(years_use)].dropna(subset=feature_cols + STATE_COLS)["year"].nunique()
    )
    countries_iso3 = sorted(complete[complete >= len(years_use) * 0.8].index.tolist())
    print(f"Countries with 80%+ coverage (1990-2025): {len(countries_iso3)}")
    print(f"Years: {years_use[0]}-{years_use[-1]} ({len(years_use)} snapshots)")
    print(f"Features: {in_dim} ({feature_cols})")

    print(f"\nBuilding graph snapshots...")
    snapshots = []
    valid_years = []
    for year in years_use:
        df_year = df[df["year"] == year]
        available = [c for c in countries_iso3 if c in df_year["country_text_id"].values]
        if len(available) < len(countries_iso3) * 0.9:
            continue
        df_year = df_year[df_year["country_text_id"].isin(countries_iso3)]
        for c in feature_cols:
            df_year[c] = df_year[c].fillna(df_year[c].median())
        for c in STATE_COLS:
            df_year[c] = df_year[c].fillna(1.0 / len(STATE_COLS))

        x, y, c_ei, a_ei, t_ei = build_snapshot(df_year, countries_iso3, mapping, feature_cols, year)

        if torch.isnan(x).any():
            continue
        snapshots.append((x, y, c_ei, a_ei, t_ei))
        valid_years.append(year)

    print(f"Built {len(snapshots)} valid snapshots")
    sample = snapshots[0]
    print(f"  Nodes: {sample[0].shape[0]}, Features: {sample[0].shape[1]}")
    print(f"  Contiguity edges: {sample[2].shape[1]}")
    print(f"  Alliance edges: {sample[3].shape[1]}")
    print(f"  Trade edges: {sample[4].shape[1]}")

    print(f"\nTraining ContagionNet...")
    model = train_model(snapshots, in_dim)

    print(f"\nComputing contagion scores...")
    scores_df = compute_contagion_scores(model, snapshots, df, countries_iso3, valid_years)

    cname_map = df.drop_duplicates("country_text_id").set_index("country_text_id")["country_name"]
    scores_df["country_name"] = scores_df["country_text_id"].map(cname_map)

    output_dir = os.path.dirname(os.path.abspath(__file__))
    scores_df.to_csv(os.path.join(output_dir, "contagion_scores.csv"), index=False)
    print(f"Saved {len(scores_df)} country-year contagion scores")

    print(f"\n=== Contagion score summary ===")
    latest = scores_df[scores_df["year"] == valid_years[-1]].sort_values("contagion_score", ascending=False)
    print(f"\nMost network-influenced (2025):")
    for _, row in latest.head(10).iterrows():
        print(f"  {row['country_name']}: contagion={row['contagion_score']:.3f}")
    print(f"\nMost domestically driven (2025):")
    for _, row in latest.tail(10).iterrows():
        print(f"  {row['country_name']}: contagion={row['contagion_score']:.3f}")

    print(f"\n=== Case studies ===")
    for country in ["Hungary", "Türkiye", "Poland", "United States of America"]:
        sub = scores_df[scores_df["country_name"] == country].sort_values("year")
        if len(sub) == 0:
            continue
        recent = sub.tail(5)
        print(f"\n{country}:")
        for _, r in recent.iterrows():
            print(f"  {int(r['year'])}: contagion={r['contagion_score']:.3f}, "
                  f"spill_F1={r['spillover_factor_1']:.4f}, spill_F2={r['spillover_factor_2']:.4f}")

    print(f"\n=== Average contagion by channel ===")
    for ch in ["contig", "alliance", "trade"]:
        cols = [c for c in scores_df.columns if f"exposure_{ch}" in c]
        if cols:
            avg = scores_df[cols].abs().mean().mean()
            print(f"  {ch}: avg |exposure| = {avg:.4f}")

    return model, scores_df


if __name__ == "__main__":
    model, scores_df = run_stage4()
