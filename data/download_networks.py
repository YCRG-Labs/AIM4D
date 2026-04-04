import os
import zipfile
import urllib.request
import pandas as pd
import networkx as nx

DATA_DIR = os.path.dirname(os.path.abspath(__file__))

COW_URL = "https://correlatesofwar.org/wp-content/uploads/DirectContiguity320.zip"
COW_ZIP = os.path.join(DATA_DIR, "contiguity.zip")
COW_DIR = os.path.join(DATA_DIR, "contiguity")

ATOP_URL = "http://www.atopdata.org/uploads/6/9/1/3/69134503/atop_5.1__.csv_.zip"
ATOP_ZIP = os.path.join(DATA_DIR, "atop.zip")
ATOP_DIR = os.path.join(DATA_DIR, "atop")


def _download_and_extract(url, zip_path, extract_dir):
    if os.path.exists(extract_dir):
        return
    os.makedirs(extract_dir, exist_ok=True)
    print(f"Downloading {url}...")
    urllib.request.urlretrieve(url, zip_path)
    with zipfile.ZipFile(zip_path, "r") as z:
        z.extractall(extract_dir)
    os.remove(zip_path)


def _find_csv(base_dir, pattern):
    for root, dirs, files in os.walk(base_dir):
        for f in files:
            if f.endswith(".csv") and pattern in f.lower():
                return os.path.join(root, f)
    for root, dirs, files in os.walk(base_dir):
        for f in files:
            if f.endswith(".csv"):
                return os.path.join(root, f)
    raise RuntimeError(f"No CSV with '{pattern}' found in {base_dir}")


def load_contiguity():
    _download_and_extract(COW_URL, COW_ZIP, COW_DIR)

    dyad_file = _find_csv(COW_DIR, "contdird")
    df = pd.read_csv(dyad_file, low_memory=False)
    print(f"COW Contiguity loaded: {len(df)} dyad-years")

    land_border = df[df["conttype"] == 1]
    G = nx.from_pandas_edgelist(
        land_border, "state1no", "state2no",
        edge_attr=["conttype", "year"],
        create_using=nx.Graph()
    )
    print(f"Contiguity graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
    return df, G


def load_atop():
    _download_and_extract(ATOP_URL, ATOP_ZIP, ATOP_DIR)

    csvs = []
    for root, dirs, files in os.walk(ATOP_DIR):
        for f in files:
            if f.endswith(".csv"):
                csvs.append(os.path.join(root, f))

    if not csvs:
        raise RuntimeError(f"No CSV found in {ATOP_DIR}")

    df = pd.read_csv(csvs[0], low_memory=False, encoding="latin-1")
    print(f"ATOP loaded: {len(df)} rows, columns: {list(df.columns[:10])}...")
    return df


if __name__ == "__main__":
    print("=== Contiguity ===")
    cont_df, cont_G = load_contiguity()

    print("\n=== ATOP Alliances ===")
    atop_df = load_atop()
    print(atop_df.head())
