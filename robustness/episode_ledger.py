"""
Machine-readable outcome ledger, reconciled against the official ERT release.

The paper's episode set has been a hand-maintained dictionary keyed by country,
which carries at most one episode per country and no episode identifiers or end
dates. This builds the ledger from the published ERT file, records every
departure the paper's set makes from it, and reports the two consequences that
bear on the forecasting results: countries with more than one autocratization
episode, and episodes beginning after the paper's latest recorded onset.

Outputs robustness/episode_ledger.csv and prints the reconciliation.
"""

import os
import sys

import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

OUT = os.path.dirname(os.path.abspath(__file__))
# ERT v16, vendored so the ledger does not depend on a path outside the repo.
# Source: https://github.com/vdeminstitute/ERT (release v16), file inst/ert.csv
ERT = os.environ.get("AIM4D_ERT_CSV",
                     os.path.join(os.path.dirname(__file__), "..", "data", "ert_v16", "ert.csv"))
ERT_SHA256 = "26e805df073a26973f5c0fbefdcdd0d20776245d71092a575063e47ffac6db5f"


def _check_source(path):
    import hashlib
    h = hashlib.sha256(open(path, "rb").read()).hexdigest()
    if h != ERT_SHA256:
        raise SystemExit("ERT source checksum mismatch: expected "
                         + ERT_SHA256 + ", got " + h)
    return h
FIRST_YEAR = 1996


def ert_episodes():
    _check_source(ERT)
    d = pd.read_csv(ERT, low_memory=False)
    e = d[d["aut_ep"] == 1]
    ep = (e.groupby(["country_name", "country_text_id", "aut_ep_id"])
            .agg(onset=("aut_ep_start_year", "min"),
                 end=("aut_ep_end_year", "max"),
                 outcome=("aut_ep_outcome_agg", "last"),
                 censored=("aut_ep_censored", "last"))
            .reset_index())
    ep = ep[ep["onset"] >= FIRST_YEAR].copy()
    ep["onset"] = ep["onset"].astype(int)
    ep["end"] = ep["end"].astype(int)
    return ep.sort_values(["country_name", "onset"])


def main():
    from stage5_ews.estimate import KNOWN_EPISODES as K
    ep = ert_episodes()
    ours = pd.DataFrame([{"country_name": c, "paper_onset": v["onset"],
                          "paper_peak": v.get("peak", v["onset"]),
                          "paper_type": v.get("type", "")} for c, v in K.items()])

    # Reconcile per paper entry. Merging on country alone matches a country with
    # three ERT episodes to its single paper row three times, which would report
    # two spurious disagreements for every recurrent country.
    by_country = {c: g for c, g in ep.groupby("country_name")}
    recs = []
    for _, r in ours.iterrows():
        g = by_country.get(r["country_name"])
        if g is None:
            recs.append({**r, "aut_ep_id": "", "onset": None, "end": None,
                         "status": "paper_addition_not_in_ert"})
            continue
        exact = g[g["onset"] == r["paper_onset"]]
        near = g[(g["onset"] - r["paper_onset"]).abs() <= 2]
        pick = exact if len(exact) else (near if len(near) else g.head(1))
        row = pick.iloc[0]
        recs.append({**r, "aut_ep_id": row["aut_ep_id"], "onset": int(row["onset"]),
                     "end": int(row["end"]),
                     "status": "onset_matches_ert" if len(exact)
                     else ("onset_within_2yr" if len(near) else "onset_differs")})
    # Carry the full ERT record onto every matched row. Building the matched rows
    # from the paper's entry alone would leave country_text_id, outcome and the
    # censoring flag empty exactly where the reconciliation is most informative.
    rec_df = pd.DataFrame(recs)
    keep = ["aut_ep_id", "country_text_id", "outcome", "censored", "end"]
    rec_df = rec_df.drop(columns=[c for c in keep if c in rec_df.columns and c != "aut_ep_id"],
                         errors="ignore")
    rec_df = rec_df.merge(ep[keep], on="aut_ep_id", how="left")
    used = set(rec_df["aut_ep_id"]) - {""}
    extra = ep[~ep["aut_ep_id"].isin(used)].copy()
    extra["status"] = "in_ert_absent_from_paper_set"
    m = pd.concat([rec_df, extra], ignore_index=True)

    cols = [c for c in ["country_name", "country_text_id", "aut_ep_id", "onset", "end",
                        "outcome", "censored", "paper_onset", "paper_peak", "paper_type",
                        "status"] if c in m.columns]
    m = m[cols].sort_values(["country_name", "onset"], na_position="last")
    dest = os.path.join(OUT, "episode_ledger.csv")
    m.to_csv(dest, index=False)

    # Assertions, because the previous version printed a success message while
    # writing nothing and left a stale country-level merge on disk.
    back = pd.read_csv(dest)
    assert len(back) == len(m), "ledger did not round-trip"
    dup = back[back["aut_ep_id"].notna()]["aut_ep_id"]
    assert dup.is_unique, f"episode ids repeat in the ledger: {dup[dup.duplicated()].tolist()[:5]}"
    assert set(back["status"]) <= {"onset_matches_ert", "onset_within_2yr", "onset_differs",
                                   "paper_addition_not_in_ert", "in_ert_absent_from_paper_set"}
    print(f"ledger written and verified: {len(back)} rows, episode ids unique")

    print(f"ERT autocratization episodes with onset >= {FIRST_YEAR}: {len(ep)} "
          f"across {ep.country_name.nunique()} countries")
    print(f"paper's hand-maintained set: {len(ours)} entries, one per country\n")

    rec = ep.groupby("country_name").size()
    multi = rec[rec > 1]
    print(f"countries ERT gives MORE THAN ONE episode: {len(multi)} "
          f"({int(multi.sum())} episodes, {int(multi.sum() - len(multi))} unrepresentable "
          f"in a country-keyed dictionary)")
    print("  " + ", ".join(f"{c} ({n})" for c, n in multi.items()))

    late = ep[ep["onset"] >= 2024]
    print(f"\nERT episodes beginning in 2024 or later: {len(late)}")
    print(late[["country_name", "aut_ep_id", "onset", "end", "censored"]].to_string(index=False))

    print("\nreconciliation against the paper's set:")
    print(m["status"].value_counts().to_string())
    print(f"\nWrote episode_ledger.csv")


if __name__ == "__main__":
    main()
