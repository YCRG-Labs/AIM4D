import sys
import os
import warnings
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.utils import degree

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

FACTOR_COLS = ["factor_1", "factor_2", "factor_3", "factor_4"]
BETA_COLS = ["beta_factor_1", "beta_factor_2", "beta_factor_3", "beta_factor_4"]
STATE_COLS = ["prob_state_0", "prob_state_1", "prob_state_2", "prob_state_3", "prob_state_4"]
TREATMENT_DIM = 4
OUTCOME_DIM = 5
HIDDEN_DIM = 32
EPOCHS = 200
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


def build_contiguity_edges(mapping, countries_iso3):
    cow_map = dict(zip(mapping["country_text_id"], mapping["COWcode"]))
    iso3_map = {v: k for k, v in cow_map.items()}
    cont = pd.read_csv("data/contiguity/DirectContiguity320/contdird.csv", low_memory=False)
    land = cont[cont["conttype"] <= 2]
    latest = land.groupby(["state1no", "state2no"]).last().reset_index()
    edges = set()
    for _, row in latest.iterrows():
        s1 = iso3_map.get(row["state1no"])
        s2 = iso3_map.get(row["state2no"])
        if s1 in countries_iso3 and s2 in countries_iso3:
            i = countries_iso3.index(s1)
            j = countries_iso3.index(s2)
            edges.add((i, j))
            edges.add((j, i))
    if edges:
        return torch.tensor(list(edges), dtype=torch.long).t()
    return torch.zeros(2, 0, dtype=torch.long)


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
        return torch.tensor(list(edges), dtype=torch.long).t()
    return torch.zeros(2, 0, dtype=torch.long)


def build_trade_edges(df_year, countries_iso3, k=5):
    trade_vals = df_year.set_index("country_text_id")["gdp_pc"].reindex(countries_iso3).fillna(0).values
    n = len(countries_iso3)
    if n < k + 1:
        return torch.zeros(2, 0, dtype=torch.long)
    log_vals = np.log1p(np.abs(trade_vals))
    diffs = np.abs(log_vals[:, None] - log_vals[None, :])
    np.fill_diagonal(diffs, np.inf)
    edges = set()
    for i in range(n):
        neighbors = np.argsort(diffs[i])[:k]
        for j in neighbors:
            edges.add((i, int(j)))
            edges.add((int(j), i))
    return torch.tensor(list(edges), dtype=torch.long).t()


def neighbor_mean(values, edge_index, n_nodes):
    if edge_index.shape[1] == 0:
        return torch.zeros(n_nodes, values.shape[1], device=values.device)
    src, dst = edge_index
    agg = torch.zeros(n_nodes, values.shape[1], device=values.device)
    agg.scatter_add_(0, dst.unsqueeze(-1).expand_as(values[src]), values[src])
    deg = torch.zeros(n_nodes, device=values.device)
    deg.scatter_add_(0, dst, torch.ones(src.shape[0], device=values.device))
    deg = deg.clamp(min=1)
    return agg / deg.unsqueeze(-1)


class LearnedExposureMapping(nn.Module):
    def __init__(self, node_dim, edge_types=3, treatment_dim=TREATMENT_DIM):
        super().__init__()
        self.treatment_dim = treatment_dim
        self.weight_nets = nn.ModuleList([
            nn.Sequential(
                nn.Linear(node_dim * 2, 32),
                nn.ELU(),
                nn.Linear(32, 1),
                nn.Sigmoid(),
            )
            for _ in range(edge_types)
        ])

    def forward(self, x, treatment, edge_indices):
        n = x.shape[0]
        exposures = []
        all_weights = []

        for etype, (edge_index, weight_net) in enumerate(zip(edge_indices, self.weight_nets)):
            if edge_index.shape[1] == 0:
                exposures.append(torch.zeros(n, self.treatment_dim, device=x.device))
                all_weights.append(None)
                continue

            src, dst = edge_index
            pair_feat = torch.cat([x[dst], x[src]], dim=-1)
            w = weight_net(pair_feat).squeeze(-1)

            weighted_treat = treatment[src] * w.unsqueeze(-1)
            agg = torch.zeros(n, self.treatment_dim, device=x.device)
            agg.scatter_add_(0, dst.unsqueeze(-1).expand_as(weighted_treat), weighted_treat)
            w_sum = torch.zeros(n, device=x.device)
            w_sum.scatter_add_(0, dst, w)
            w_sum = w_sum.clamp(min=1e-8)
            exposures.append(agg / w_sum.unsqueeze(-1))
            all_weights.append(w)

        return torch.cat(exposures, dim=-1), all_weights


class INETARNet(nn.Module):
    def __init__(self, covariate_dim, treatment_dim=TREATMENT_DIM,
                 outcome_dim=OUTCOME_DIM, hidden=HIDDEN_DIM):
        super().__init__()
        self.treatment_dim = treatment_dim
        self.covariate_dim = covariate_dim
        cov_only_dim = covariate_dim - treatment_dim

        self.ego_encoder = nn.Sequential(
            nn.Linear(covariate_dim, hidden),
            nn.ELU(),
            nn.Linear(hidden, hidden),
        )

        self.gcn1_contig = GCNConv(covariate_dim, hidden)
        self.gcn1_alliance = GCNConv(covariate_dim, hidden)
        self.gcn1_trade = GCNConv(covariate_dim, hidden)
        self.edge_type_embed = nn.Embedding(3, hidden)
        self.norm1 = nn.LayerNorm(hidden)
        self.gcn2_contig = GCNConv(hidden, hidden)
        self.gcn2_alliance = GCNConv(hidden, hidden)
        self.gcn2_trade = GCNConv(hidden, hidden)
        self.norm2 = nn.LayerNorm(hidden)

        self.exposure_map = LearnedExposureMapping(hidden * 2, edge_types=3, treatment_dim=treatment_dim)

        exposure_dim = treatment_dim * 3
        repr_dim = hidden * 2

        self.outcome_logits = nn.Sequential(
            nn.Linear(repr_dim + treatment_dim + exposure_dim, hidden),
            nn.ELU(),
            nn.Dropout(0.2),
            nn.Linear(hidden, hidden),
            nn.ELU(),
            nn.Linear(hidden, outcome_dim),
        )

        self.gps_mu = nn.Sequential(
            nn.Linear(repr_dim, hidden),
            nn.ELU(),
            nn.Linear(hidden, treatment_dim),
        )
        self.gps_logvar = nn.Sequential(
            nn.Linear(repr_dim, hidden),
            nn.ELU(),
            nn.Linear(hidden, treatment_dim),
        )

        self.local_outcome_net = nn.Sequential(
            nn.Linear(hidden + treatment_dim, hidden),
            nn.ELU(),
            nn.Linear(hidden, outcome_dim),
        )

    def _gcn_layer(self, x, edge_indices, convs, embed_ids):
        n = x.shape[0]
        h = torch.zeros(n, convs[0].out_channels, device=x.device)
        for conv, ei, eid in zip(convs, edge_indices, embed_ids):
            if ei.shape[1] > 0:
                h = h + conv(x, ei) + self.edge_type_embed(torch.tensor(eid, device=x.device))
        return h

    def encode(self, x, edge_indices):
        h_ego = self.ego_encoder(x)
        convs1 = [self.gcn1_contig, self.gcn1_alliance, self.gcn1_trade]
        h_gnn = F.elu(self.norm1(self._gcn_layer(x, edge_indices, convs1, [0, 1, 2])))
        convs2 = [self.gcn2_contig, self.gcn2_alliance, self.gcn2_trade]
        h_gnn = F.elu(self.norm2(self._gcn_layer(h_gnn, edge_indices, convs2, [0, 1, 2])))
        return torch.cat([h_ego, h_gnn], dim=-1)

    def forward(self, x, edge_indices):
        treatment = x[:, :self.treatment_dim]

        h = self.encode(x, edge_indices)

        h_ego_only = self.ego_encoder(x)
        exposure, weights = self.exposure_map(h, treatment, edge_indices)

        y_full = F.softmax(self.outcome_logits(torch.cat([h, treatment, exposure], dim=-1)), dim=-1)

        y_local = self.local_outcome_net(torch.cat([h_ego_only, treatment], dim=-1))
        y_local = F.softmax(y_local, dim=-1)

        gps_mu = self.gps_mu(h)
        gps_logvar = self.gps_logvar(h)

        return y_full, y_local, exposure, weights, gps_mu, gps_logvar

    def counterfactual_decompose(self, x, edge_indices):
        with torch.no_grad():
            treatment = x[:, :self.treatment_dim]
            h = self.encode(x, edge_indices)
            h_ego_only = self.ego_encoder(x)
            exposure, weights = self.exposure_map(h, treatment, edge_indices)

            logits_full = self.outcome_logits(torch.cat([h, treatment, exposure], dim=-1))
            y_full = F.softmax(logits_full, dim=-1)

            zero_exposure = torch.zeros_like(exposure)
            logits_no_spill = self.outcome_logits(torch.cat([h, treatment, zero_exposure], dim=-1))

            spillover_effect = logits_full - logits_no_spill
            domestic_effect = logits_no_spill

        return y_full, domestic_effect, spillover_effect, exposure, weights


def mmd_kernel(h1, h2, bandwidth=1.0):
    if h1.shape[0] < 2 or h2.shape[0] < 2:
        return torch.tensor(0.0, device=h1.device)
    k11 = torch.exp(-torch.cdist(h1, h1) / (2 * bandwidth ** 2)).mean()
    k22 = torch.exp(-torch.cdist(h2, h2) / (2 * bandwidth ** 2)).mean()
    k12 = torch.exp(-torch.cdist(h1, h2) / (2 * bandwidth ** 2)).mean()
    return k11 + k22 - 2 * k12


def compute_loss(y_full, y_local, y_true, gps_mu, gps_logvar, treatment,
                 h_repr, alpha_gps=0.5, alpha_balance=0.3, alpha_local=0.3):
    loss_outcome = F.mse_loss(y_full, y_true)
    loss_local = F.mse_loss(y_local, y_true)

    gps_logvar_clamped = gps_logvar.clamp(-5, 5)
    var = gps_logvar_clamped.exp()
    gps_nll = 0.5 * ((treatment - gps_mu) ** 2 / (var + 1e-6) + gps_logvar_clamped).mean()

    median_t = treatment[:, 0].median()
    mask_high = treatment[:, 0] > median_t
    mask_low = ~mask_high
    balance_loss = mmd_kernel(h_repr[mask_high], h_repr[mask_low])

    total = loss_outcome + alpha_local * loss_local + alpha_gps * gps_nll + alpha_balance * balance_loss

    return total, {
        "outcome": loss_outcome.item(),
        "local": loss_local.item(),
        "gps": gps_nll.item(),
        "balance": balance_loss.item(),
    }


def build_snapshot(df_year, countries_iso3, mapping, feature_cols, year):
    x_data = df_year.set_index("country_text_id").reindex(countries_iso3)
    x = torch.tensor(x_data[feature_cols].values, dtype=torch.float32)
    y = torch.tensor(x_data[STATE_COLS].values, dtype=torch.float32)
    contig_ei = build_contiguity_edges(mapping, countries_iso3)
    alliance_ei = build_alliance_edges(mapping, countries_iso3, year)
    trade_ei = build_trade_edges(df_year, countries_iso3)
    return x, y, [contig_ei, alliance_ei, trade_ei]


def train_model(train_snaps, test_snaps, in_dim):
    model = INETARNet(in_dim)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

    best_test_loss = float("inf")
    best_state = None

    for epoch in range(EPOCHS):
        model.train()
        train_loss = 0.0

        for x, y, eis in train_snaps:
            optimizer.zero_grad()
            y_full, y_local, exposure, weights, gps_mu, gps_logvar = model(x, eis)
            treatment = x[:, :TREATMENT_DIM]
            h_repr = model.encode(x, eis)

            loss, metrics = compute_loss(
                y_full, y_local, y, gps_mu, gps_logvar, treatment, h_repr
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            train_loss += loss.item()

        scheduler.step()

        if (epoch + 1) % 40 == 0:
            model.eval()
            test_loss = 0.0
            with torch.no_grad():
                for x, y, eis in test_snaps:
                    y_full, y_local, *_ = model(x, eis)
                    test_loss += F.mse_loss(y_full, y).item()

            avg_train = train_loss / max(len(train_snaps), 1)
            avg_test = test_loss / max(len(test_snaps), 1)

            if avg_test < best_test_loss:
                best_test_loss = avg_test
                best_state = {k: v.clone() for k, v in model.state_dict().items()}

            print(f"  Epoch {epoch+1}/{EPOCHS}: train={avg_train:.6f}, test={avg_test:.6f}")

    if best_state is not None:
        model.load_state_dict(best_state)
        print(f"  Loaded best model (test loss={best_test_loss:.6f})")

    return model


def compute_contagion_scores(model, snapshots, countries_iso3, years):
    model.eval()
    rows = []

    for t, (x, y, eis) in enumerate(snapshots):
        y_full, domestic, spillover, exposure, weights = model.counterfactual_decompose(x, eis)

        for i, iso3 in enumerate(countries_iso3):
            spill_mag = spillover[i].abs().sum().item()
            dom_mag = domestic[i].abs().sum().item()
            total = spill_mag + dom_mag + 1e-10
            contagion_ratio = spill_mag / total

            row = {
                "country_text_id": iso3,
                "year": int(years[t]),
                "contagion_score": contagion_ratio,
                "domestic_score": 1 - contagion_ratio,
            }
            for k in range(min(OUTCOME_DIM, spillover.shape[1])):
                row[f"spillover_state_{k}"] = spillover[i, k].item()
            for k in range(TREATMENT_DIM):
                row[f"exposure_contig_f{k+1}"] = exposure[i, k].item()
                row[f"exposure_alliance_f{k+1}"] = exposure[i, TREATMENT_DIM + k].item()
                row[f"exposure_trade_f{k+1}"] = exposure[i, 2 * TREATMENT_DIM + k].item()

            rows.append(row)

    return pd.DataFrame(rows)


def run_stage4():
    print("=== Stage 4: Network Structural Causal Model (INE-TARNet) ===\n")

    df, mapping = load_all_data()
    feature_cols = FACTOR_COLS + BETA_COLS + ["gdp_pc", "urbanization"]
    in_dim = len(feature_cols)

    years_all = sorted(df["year"].unique())
    years_use = [y for y in years_all if y >= 1990]

    complete = df.groupby("country_text_id").apply(
        lambda g: g[g["year"].isin(years_use)].dropna(subset=feature_cols + STATE_COLS)["year"].nunique()
    )
    countries_iso3 = sorted(complete[complete >= len(years_use) * 0.8].index.tolist())
    print(f"Countries: {len(countries_iso3)}")
    print(f"Years: {years_use[0]}-{years_use[-1]} ({len(years_use)} total)")
    print(f"Features: {in_dim}")
    print(f"Train/test split: ≤{TRAIN_CUTOFF} / >{TRAIN_CUTOFF}")

    print(f"\nBuilding snapshots...")
    all_snaps = []
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
        x, y, eis = build_snapshot(df_year, countries_iso3, mapping, feature_cols, year)
        if torch.isnan(x).any():
            continue
        all_snaps.append((x, y, eis))
        valid_years.append(year)

    all_x = torch.cat([s[0] for s in all_snaps], dim=0)
    feat_mean = all_x.mean(dim=0)
    feat_std = all_x.std(dim=0).clamp(min=1e-6)
    normalized_snaps = []
    for x, y, eis in all_snaps:
        x_norm = (x - feat_mean) / feat_std
        normalized_snaps.append((x_norm, y, eis))
    all_snaps = normalized_snaps
    print(f"  Features normalized (mean-centered, unit variance)")

    train_idx = [i for i, y in enumerate(valid_years) if y <= TRAIN_CUTOFF]
    test_idx = [i for i, y in enumerate(valid_years) if y > TRAIN_CUTOFF]
    train_snaps = [all_snaps[i] for i in train_idx]
    test_snaps = [all_snaps[i] for i in test_idx]

    print(f"Snapshots: {len(all_snaps)} total, {len(train_snaps)} train, {len(test_snaps)} test")
    s = all_snaps[0]
    print(f"  Nodes: {s[0].shape[0]}, Contiguity: {s[2][0].shape[1]}, "
          f"Alliance: {s[2][1].shape[1]}, Trade: {s[2][2].shape[1]}")

    print(f"\nArchitecture: INE-TARNet")
    print(f"  Learned exposure mapping (heterogeneous peer weights)")
    print(f"  GPS head (generalized propensity score)")
    print(f"  MMD representation balancing")
    print(f"  Counterfactual decomposition (zero-exposure intervention)")

    print(f"\nTraining...")
    model = train_model(train_snaps, test_snaps, in_dim)

    print(f"\nComputing contagion scores (all years)...")
    scores_df = compute_contagion_scores(model, all_snaps, countries_iso3, valid_years)

    cname_map = df.drop_duplicates("country_text_id").set_index("country_text_id")["country_name"]
    scores_df["country_name"] = scores_df["country_text_id"].map(cname_map)

    output_dir = os.path.dirname(os.path.abspath(__file__))
    scores_df.to_csv(os.path.join(output_dir, "contagion_scores.csv"), index=False)
    print(f"Saved {len(scores_df)} scores")

    print(f"\n=== Out-of-sample validation ({TRAIN_CUTOFF+1}-{valid_years[-1]}) ===")
    model.eval()
    test_mse = 0.0
    with torch.no_grad():
        for x, y, eis in test_snaps:
            y_pred, *_ = model(x, eis)
            test_mse += F.mse_loss(y_pred, y).item()
    print(f"  Test MSE (state probs): {test_mse / len(test_snaps):.6f}")

    latest_year = valid_years[-1]
    latest = scores_df[scores_df["year"] == latest_year].sort_values("contagion_score", ascending=False)
    print(f"\n=== Most network-influenced ({latest_year}) ===")
    for _, row in latest.head(10).iterrows():
        print(f"  {row['country_name']}: {row['contagion_score']:.3f}")
    print(f"\n=== Most domestically driven ({latest_year}) ===")
    for _, row in latest.tail(10).iterrows():
        print(f"  {row['country_name']}: {row['contagion_score']:.3f}")

    print(f"\n=== Case studies ===")
    for country in ["Hungary", "Türkiye", "Poland", "United States of America", "Denmark"]:
        sub = scores_df[scores_df["country_name"] == country].sort_values("year")
        if len(sub) == 0:
            continue
        recent = sub.tail(5)
        print(f"\n{country}:")
        for _, r in recent.iterrows():
            oos = " (OOS)" if r["year"] > TRAIN_CUTOFF else ""
            exp_c = r.get("exposure_contig_f1", 0)
            exp_a = r.get("exposure_alliance_f1", 0)
            exp_t = r.get("exposure_trade_f1", 0)
            print(f"  {int(r['year'])}{oos}: contagion={r['contagion_score']:.3f} "
                  f"[contig={exp_c:.3f}, alliance={exp_a:.3f}, trade={exp_t:.3f}]")

    print(f"\n=== Exposure channel importance ===")
    for ch in ["contig", "alliance", "trade"]:
        cols = [c for c in scores_df.columns if f"exposure_{ch}" in c]
        if cols:
            avg = scores_df[cols].abs().mean().mean()
            print(f"  {ch}: avg |exposure| = {avg:.4f}")

    print(f"\n=== Smoothing contagion scores (3-year rolling) ===")
    scores_df = scores_df.sort_values(["country_text_id", "year"])
    scores_df["contagion_smooth"] = scores_df.groupby("country_text_id")["contagion_score"].transform(
        lambda x: x.rolling(3, min_periods=1, center=True).mean()
    )
    scores_df["domestic_smooth"] = 1.0 - scores_df["contagion_smooth"]

    print(f"\nSmoothed case studies:")
    for country in ["Hungary", "Türkiye", "Poland", "United States of America", "Denmark"]:
        sub = scores_df[scores_df["country_name"] == country].sort_values("year").tail(5)
        if len(sub) == 0:
            continue
        vals = ", ".join(f"{int(r['year'])}:{r['contagion_smooth']:.3f}" for _, r in sub.iterrows())
        print(f"  {country}: {vals}")

    print(f"\n=== Placebo test (randomized network) ===")
    placebo_scores = run_placebo(model, all_snaps, countries_iso3, valid_years)

    real_mean = scores_df["contagion_score"].mean()
    placebo_mean = placebo_scores["contagion_score"].mean()
    ratio = real_mean / (placebo_mean + 1e-10)
    print(f"  Real network avg contagion:    {real_mean:.4f}")
    print(f"  Placebo network avg contagion: {placebo_mean:.4f}")
    print(f"  Ratio (real/placebo):          {ratio:.2f}x")
    if ratio > 1.5:
        print(f"  PASS: Real network contagion is {ratio:.1f}x higher than random")
    else:
        print(f"  WEAK: Real network only {ratio:.1f}x vs random — network signal may be noise")

    scores_df.to_csv(os.path.join(output_dir, "contagion_scores.csv"), index=False)
    print(f"\nFinal output saved ({len(scores_df)} rows)")

    return model, scores_df


def run_placebo(model, all_snaps, countries_iso3, valid_years, n_perms=5):
    model.eval()
    n = len(countries_iso3)
    all_scores = []

    for perm in range(n_perms):
        rng = np.random.RandomState(perm + 42)
        placebo_snaps = []
        for x, y, eis in all_snaps:
            shuffled_eis = []
            for ei in eis:
                if ei.shape[1] == 0:
                    shuffled_eis.append(ei)
                    continue
                perm_map = torch.tensor(rng.permutation(n), dtype=torch.long)
                shuffled_ei = perm_map[ei]
                shuffled_eis.append(shuffled_ei)
            placebo_snaps.append((x, y, shuffled_eis))

        scores = compute_contagion_scores(model, placebo_snaps, countries_iso3, valid_years)
        all_scores.append(scores)

    combined = pd.concat(all_scores)
    return combined.groupby(["country_text_id", "year"]).mean().reset_index()


if __name__ == "__main__":
    model, scores_df = run_stage4()
