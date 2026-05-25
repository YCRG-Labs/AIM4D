"""Leave-one-neighbor-out for the contagion channel: confirm the recent-era
cultural-bloc diffusion signal does not hinge on a single influential hub
(Hungary in particular). For each high-degree country we zero its edges in the
cultural weight matrix, rebuild the spatial lag, and re-estimate the post-2005
change-spec coefficient. A stable alpha across drops means no one node drives it.
"""

import os
import sys
import warnings

warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from contagion_fix import build_inputs, build_W_cultural, build_panel, clustered

OUTPUT_DIR = os.path.dirname(os.path.abspath(__file__))
SPLIT_YEAR = 2005


def post2005_alpha(df, countries, years, Y, W):
    p = build_panel(df, countries, years, Y, W)
    sub = p[p["year"] >= SPLIT_YEAR]
    if sub["state"].nunique() < 2 or len(sub) < 50:
        return np.nan, np.nan
    m = clustered("dy ~ y_lag + wdy + C(state)", sub)
    return m.params["wdy"], m.pvalues["wdy"]


def drop_node(W, countries, name):
    W2 = W.copy()
    if name in countries:
        i = countries.index(name)
        W2[i, :] = 0.0
        W2[:, i] = 0.0
        rs = W2.sum(axis=1, keepdims=True)
        rs[rs == 0] = 1
        W2 = W2 / rs
    return W2


def main():
    print("=" * 70)
    print("CONTAGION LEAVE-ONE-NEIGHBOR-OUT (cultural W, post-2005)")
    print("=" * 70)
    df, countries, years, Y = build_inputs()
    W = build_W_cultural(countries)

    a0, p0 = post2005_alpha(df, countries, years, Y, W)
    print(f"\n  full cultural W:        alpha={a0:+.4f}  p={p0:.3f}")

    degree = W.sum(axis=1)
    top = [countries[i] for i in np.argsort(-degree)[:6]]
    hubs = (["HUN"] if "HUN" in countries else []) + [c for c in top if c != "HUN"]
    out = {"full_alpha": a0, "full_p": p0}
    alphas = [a0]
    print("\n  leave-one-out (drop each hub's edges):")
    for name in hubs[:6]:
        a, p = post2005_alpha(df, countries, years, Y, drop_node(W, countries, name))
        tag = " (Hungary hub)" if name == "HUN" else ""
        print(f"    drop {name:5s}: alpha={a:+.4f}  p={p:.3f}{tag}")
        out[f"drop_{name}_alpha"] = a
        alphas.append(a)
    rng = (min(alphas), max(alphas))
    print(f"\n  alpha range across drops: [{rng[0]:+.4f}, {rng[1]:+.4f}]  "
          f"({'STABLE: no single hub drives it' if rng[1] - rng[0] < 0.1 else 'sensitive to a hub'})")
    out["alpha_min"] = rng[0]
    out["alpha_max"] = rng[1]

    pd.DataFrame([out]).to_csv(os.path.join(OUTPUT_DIR, "contagion_lono_results.csv"), index=False)
    print(f"\nSaved to robustness/contagion_lono_results.csv")


if __name__ == "__main__":
    main()
