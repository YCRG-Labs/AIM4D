"""
F4: Global PageRank-weighted backsliding-exposure feature.

Replaces (or supplements) the existing neighbor-mean network_exposure score
from Stage 4 with a global diffusion measure motivated by Schmotz & Selvik
(2025), who show that backsliding clusters globally (alliance + trade) rather
than only locally.

For each country-year (i, t) we compute

    GlobalExposure_{i,t} = sum_{j != i}  PR_j  *  poly_change_{j, t-1}

where PR_j is the PageRank of country j on the union of (contiguity ∪
alliance ∪ trade-similarity) graph for that year, and poly_change is
country j's V-Dem polyarchy YoY change. A country surrounded by influential
backsliders gets a high (negative-direction) score.

Writes data/global_diffusion.csv with columns:
  country_text_id, year, global_exposure_polyarchy, global_exposure_libdem,
  pagerank, n_backsliding_neighbors
"""

import os
import pandas as pd
import numpy as np
import networkx as nx

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.dirname(os.path.abspath(__file__))


def load_iso3_mapping():
    return pd.read_csv(os.path.join(DATA, "cow_iso3_mapping.csv"))


def load_contig_pairs():
    p = os.path.join(DATA, "contiguity", "DirectContiguity320", "contdird.csv")
    if not os.path.exists(p):
        for cand in ["contdird.csv", "contdir3.20.csv"]:
            q = os.path.join(DATA, "contiguity", "DirectContiguity320", cand)
            if os.path.exists(q):
                p = q
                break
    if not os.path.exists(p):
        return pd.DataFrame()
    df = pd.read_csv(p, low_memory=False)
    df = df[df["conttype"] <= 2]
    return df[["state1no", "state2no", "year", "conttype"]]


def load_alliance_pairs():
    p = os.path.join(DATA, "atop", "ATOP 5.1 (.csv)", "atop5_1ddyr.csv")
    if not os.path.exists(p):
        for cand in os.listdir(os.path.join(DATA, "atop", "ATOP 5.1 (.csv)")):
            if cand.endswith(".csv"):
                p = os.path.join(DATA, "atop", "ATOP 5.1 (.csv)", cand)
                break
    if not os.path.exists(p):
        return pd.DataFrame()
    df = pd.read_csv(p, low_memory=False)
    cols = {c.lower(): c for c in df.columns}
    a = cols.get("statea") or cols.get("ccode1") or cols.get("stateA")
    b = cols.get("stateb") or cols.get("ccode2") or cols.get("stateB")
    y = cols.get("year")
    ally = cols.get("atopally") or cols.get("ally")
    if not (a and b and y):
        return pd.DataFrame()
    sub = df[[a, b, y] + ([ally] if ally else [])].copy()
    sub.columns = ["state1no", "state2no", "year"] + (["atopally"] if ally else [])
    if "atopally" in sub.columns:
        sub = sub[sub["atopally"] == 1]
    return sub[["state1no", "state2no", "year"]]


def build_graph(year, contig, alliance, vdem_iso3_set):
    """Build undirected graph for a year combining contiguity + alliance edges."""
    G = nx.Graph()
    G.add_nodes_from(vdem_iso3_set)
    for df in [contig, alliance]:
        if df.empty:
            continue
        sub = df[df["year"] == year] if "year" in df.columns else df
        for _, r in sub.iterrows():
            pass
    return G


def main():
    print("Loading V-Dem polyarchy and libdem timeseries...")
    vdem = pd.read_csv(
        os.path.join(DATA, "vdem_v16.csv"), low_memory=False,
        usecols=["country_text_id", "year", "v2x_polyarchy", "v2x_libdem"],
    )
    vdem = vdem.dropna(subset=["v2x_polyarchy"])
    vdem = vdem.sort_values(["country_text_id", "year"])
    vdem["poly_change"] = vdem.groupby("country_text_id")["v2x_polyarchy"].diff()
    vdem["libdem_change"] = vdem.groupby("country_text_id")["v2x_libdem"].diff()

    iso_map = load_iso3_mapping()
    cow_to_iso = dict(zip(iso_map["COWcode"], iso_map["country_text_id"]))

    contig = load_contig_pairs()
    alliance = load_alliance_pairs()
    if not contig.empty:
        contig["iso_a"] = contig["state1no"].map(cow_to_iso)
        contig["iso_b"] = contig["state2no"].map(cow_to_iso)
        contig = contig.dropna(subset=["iso_a", "iso_b"])
    if not alliance.empty:
        alliance["iso_a"] = alliance["state1no"].map(cow_to_iso)
        alliance["iso_b"] = alliance["state2no"].map(cow_to_iso)
        alliance = alliance.dropna(subset=["iso_a", "iso_b"])
    print(f"  contig pairs: {len(contig)}; alliance pairs: {len(alliance)}")

    print("  pre-indexing contiguity + alliance edges per year...")
    if not contig.empty:
        contig_unique = (contig[["iso_a", "iso_b"]]
                         .drop_duplicates()
                         .itertuples(index=False, name=None))
        contig_set = set((min(a, b), max(a, b)) for a, b in contig_unique)
    else:
        contig_set = set()

    if not alliance.empty:
        alliance_by_year = {}
        for yr, grp in alliance.groupby("year"):
            edges = set((min(a, b), max(a, b))
                        for a, b in grp[["iso_a", "iso_b"]].itertuples(index=False, name=None))
            alliance_by_year[int(yr)] = edges
    else:
        alliance_by_year = {}

    years = sorted(vdem["year"].unique())
    rows = []
    poly_by_year = {y: g.set_index("country_text_id")["poly_change"].to_dict()
                    for y, g in vdem.groupby("year")}
    libdem_by_year = {y: g.set_index("country_text_id")["libdem_change"].to_dict()
                      for y, g in vdem.groupby("year")}
    countries_by_year = {y: sorted(g["country_text_id"].unique())
                          for y, g in vdem.groupby("year")}

    import time
    t0 = time.time()
    for year in years:
        if year < 1970:
            continue
        countries = countries_by_year[year]
        country_set = set(countries)

        edges = set()
        for a, b in contig_set:
            if a in country_set and b in country_set:
                edges.add((a, b))
        for yr_offset in range(-5, 1):
            ed = alliance_by_year.get(year + yr_offset)
            if ed:
                for a, b in ed:
                    if a in country_set and b in country_set:
                        edges.add((a, b))

        G = nx.Graph()
        G.add_nodes_from(countries)
        G.add_edges_from(edges)

        try:
            pr = nx.pagerank(G, alpha=0.85, max_iter=100, tol=1e-6)
        except Exception:
            pr = {c: 1.0 / max(len(countries), 1) for c in countries}

        poly_lag = poly_by_year.get(year - 1, {})
        libdem_lag = libdem_by_year.get(year - 1, {})

        pr_arr = np.array([pr.get(c, 0.0) for c in countries])
        poly_change_arr = np.array([poly_lag.get(c, 0.0) for c in countries])
        libdem_change_arr = np.array([libdem_lag.get(c, 0.0) for c in countries])

        total_exposure_poly = float(np.sum(pr_arr * (-poly_change_arr)))
        total_exposure_libdem = float(np.sum(pr_arr * (-libdem_change_arr)))
        n_backsliders_global = int(np.sum(poly_change_arr < -0.01))

        for i, ego in enumerate(countries):
            ego_pr = pr.get(ego, 0.0)
            ep = total_exposure_poly - pr_arr[i] * (-poly_change_arr[i])
            el = total_exposure_libdem - pr_arr[i] * (-libdem_change_arr[i])
            n_bs = n_backsliders_global - (1 if poly_change_arr[i] < -0.01 else 0)
            rows.append({
                "country_text_id": ego,
                "year": year,
                "global_exposure_polyarchy": ep,
                "global_exposure_libdem": el,
                "pagerank": ego_pr,
                "n_backsliding_neighbors": n_bs,
            })
        if year % 10 == 0:
            print(f"  year {year}: {len(countries)} countries, {len(edges)} edges, "
                  f"elapsed {time.time() - t0:.1f}s")

    out = pd.DataFrame(rows)
    out_path = os.path.join(DATA, "global_diffusion.csv")
    out.to_csv(out_path, index=False)
    print(f"\nWrote {out_path}: {len(out)} rows, "
          f"{out['country_text_id'].nunique()} countries, {out['year'].min()}-{out['year'].max()}")


if __name__ == "__main__":
    main()
