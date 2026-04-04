import os
import subprocess
import pandas as pd

DATA_DIR = os.path.dirname(os.path.abspath(__file__))
CSV_PATH = os.path.join(DATA_DIR, "vdem_v16.csv")


def download_vdem():
    if os.path.exists(CSV_PATH):
        print(f"V-Dem CSV already exists at {CSV_PATH}")
        return CSV_PATH

    print("Exporting V-Dem v16 via R vdemdata package...")
    subprocess.run([
        "Rscript", "-e",
        f'library(vdemdata); write.csv(vdem, "{CSV_PATH}", row.names=FALSE)'
    ], check=True)
    print(f"V-Dem saved to {CSV_PATH}")
    return CSV_PATH


def load_vdem(path=None):
    if path is None:
        path = CSV_PATH
    if not os.path.exists(path):
        path = download_vdem()

    df = pd.read_csv(path, low_memory=False)
    print(f"V-Dem loaded: {df.shape[0]} rows, {df.shape[1]} columns")
    print(f"Countries: {df['country_name'].nunique()}, Years: {df['year'].min()}-{df['year'].max()}")
    return df


if __name__ == "__main__":
    df = load_vdem()
    print(df[["country_name", "year", "v2x_polyarchy", "v2x_libdem"]].head(10))
