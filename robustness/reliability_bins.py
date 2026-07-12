"""
Persist the Figure 7 reliability-diagram bins: five equal-width predicted-risk
bins on the strict OOS slice (year > 2019, post-onset country-years excluded),
each with mean predicted risk, observed onset frequency, and bin count.

Output: robustness/reliability_bins.csv
"""

import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from stage5_ews.estimate import KNOWN_EPISODES

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                   "reliability_bins.csv")
LEAD = 5


def main():
    ews = pd.read_csv(os.path.join(REPO, "stage5_ews", "ews_signals.csv"))
    pre, post = set(), set()
    for c, info in KNOWN_EPISODES.items():
        onset = info["onset"]
        for y in range(onset - LEAD, onset + 1):
            pre.add((c, y))
        for y in range(onset + 1, onset + 6):
            post.add((c, y))
    ews["lbl"] = [1 if (c, y) in pre else 0
                  for c, y in zip(ews["country_name"], ews["year"])]
    ews["post"] = [(c, y) in post
                   for c, y in zip(ews["country_name"], ews["year"])]

    oos = ews[(ews["year"] > 2019) & (~ews["post"])
              & ews["combined_risk"].notna()]
    s = oos["combined_risk"].values
    y = oos["lbl"].values

    edges = np.linspace(s.min(), s.max(), 6)
    which = np.digitize(s, edges[1:-1])
    rows = []
    for b in range(5):
        m = which == b
        if m.sum() == 0:
            continue
        rows.append({"bin": b + 1,
                     "mean_predicted": float(s[m].mean()),
                     "observed_freq": float(y[m].mean()),
                     "n": int(m.sum())})
    out = pd.DataFrame(rows)
    out.to_csv(OUT, index=False)
    print(out.round(3).to_string(index=False))
    print(f"Wrote {OUT}  (n_total={len(oos)})")


if __name__ == "__main__":
    main()
