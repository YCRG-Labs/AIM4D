"""Shared ERT v16 panel builder.

The hand-maintained ledger in stage5_ews.estimate.KNOWN_EPISODES is keyed by
country and so carries at most one episode per country. ERT v16 records
recurrent onsets for twenty countries, which a country-keyed structure cannot
represent. Every analysis that wants the published outcome rather than ours
builds its panel here, so the episode handling is written once.

Two things differ from onset_forecast_clean.build_panel():

  * in_episode is the union of all ERT intervals for a country, not the single
    onset-to-peak span of its one ledger entry.
  * labels are computed against the full onset list, so a country-year is
    positive when ANY onset falls in its window. Under the ledger a second
    onset after an earlier episode closed was silently unlabelled.

The upstream features are untouched and label-free (Stage 1-4 are unsupervised
and the engineered Stage 5 columns derive from V-Dem series, not from episode
coding), so swapping the outcome here changes the outcome and nothing else.
"""

import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from episode_ledger import ert_episodes

OUT = os.path.dirname(os.path.abspath(__file__))


def build_panel_ert():
    """The Stage 5 feature panel with ERT v16 episode bookkeeping attached."""
    d = pd.read_csv(os.path.join(OUT, "..", "stage5_ews", "ews_signals.csv"))
    v = pd.read_csv(os.path.join(OUT, "..", "data", "vdem_v16.csv"), low_memory=False,
                    usecols=["country_name", "year", "v2x_regime"]).dropna(subset=["v2x_regime"])
    v["v2x_regime"] = v["v2x_regime"].astype(int)
    d = d.merge(v, on=["country_name", "year"], how="left")

    ep = ert_episodes()
    spans, onsets = {}, {}
    for _, r in ep.iterrows():
        spans.setdefault(r["country_name"], []).append((int(r["onset"]), int(r["end"])))
        onsets.setdefault(r["country_name"], []).append(int(r["onset"]))

    cn, yr = d["country_name"].values, d["year"].values
    in_ep = np.zeros(len(d), bool)
    for i in range(len(d)):
        for a, b in spans.get(cn[i], ()):
            if a <= yr[i] <= b:
                in_ep[i] = True
                break
    d = d.copy()
    d["in_episode"] = in_ep
    d["at_risk"] = (d["v2x_regime"] >= 2) & (~d["in_episode"])
    d.attrs["onsets"] = onsets
    return d


def label_ert(d, h, future_only=True):
    """Positive when any ERT onset falls in the window.

    future_only selects t+1..t+h (a forecast); otherwise t..t+h (a window that
    includes the predictor year), which is the contrast the factorial varies.
    """
    onsets = d.attrs["onsets"]
    lo = 1 if future_only else 0
    cn, yr = d["country_name"].values, d["year"].values
    y = np.zeros(len(d), int)
    for i in range(len(d)):
        for a in onsets.get(cn[i], ()):
            if lo <= a - yr[i] <= h:
                y[i] = 1
                break
    return pd.Series(y, index=d.index)
