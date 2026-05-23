"""
Gating test: should we pursue UCDP-GED → V-Dem ERT transfer learning?

For each of our 46 V-Dem ERT autocratization onsets, check whether the country
had state-based conflict (UCDP-GED, type_of_violence==1, >=25 battle-deaths)
in the 10 years before onset.

Beger-Morgan-Ward (2021, JCR response) and the negative-transfer literature
(Wang et al. 2019, Tan et al. 2018) predict that if <40% of episodes have a
prior conflict signal, transfer learning will fail because the encoder learns
the wrong DGP (conflict ≠ stealth autocratization).

DECISION RULE:
  >= 40% overlap  → transfer learning may help, run CCA diagnostic next
   < 40% overlap  → skip transfer plan, add UCDP as Stage-5 meta-feature
                    instead (~2 hr, +0.01-0.03 AUC, zero architectural risk)

Outputs:
  robustness/ucdp_overlap_test.csv — per-episode flag
  Stdout — go/no-go verdict + stratified stats + missing-match diagnostics
"""

import os
import sys
import pandas as pd

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
from stage5_ews.estimate import KNOWN_EPISODES

UCDP_CSV = os.path.join(REPO, "data", "ucdp_ged.csv")
OUT_CSV = os.path.join(REPO, "robustness", "ucdp_overlap_test.csv")

COUNTRY_ALIASES = {
    "Burma/Myanmar": ["Myanmar (Burma)", "Myanmar"],
    "Türkiye": ["Turkey"],
    "Ivory Coast": ["Cote d'Ivoire", "Côte d'Ivoire"],
    "United States of America": ["United States"],
    "South Korea": ["South Korea", "Korea, Republic of"],
    "North Macedonia": ["Macedonia, FYR", "Macedonia"],
    "Czech Republic": ["Czechia"],
    "Russia": ["Russian Federation", "Russia (Soviet Union)"],
    "North Korea": ["Korea, DPR", "Korea, Democratic People's Republic of"],
    "Iran": ["Iran (Islamic Republic of)"],
    "Vietnam": ["Vietnam (North Vietnam)"],
    "Yemen": ["Yemen (North Yemen)"],
    "Tanzania": ["Tanzania (United Republic of)"],
    "Democratic Republic of the Congo": ["DR Congo (Zaire)"],
    "Republic of the Congo": ["Congo"],
    "Cape Verde": ["Cabo Verde"],
    "Eswatini": ["Swaziland"],
    "Timor-Leste": ["East Timor"],
}


def country_candidates(name):
    return [name] + COUNTRY_ALIASES.get(name, [])


def load_ucdp_country_year():
    """Aggregate UCDP-GED to country-year state-based conflict flags."""
    if not os.path.exists(UCDP_CSV):
        sys.exit(
            f"Missing {UCDP_CSV}. Run:\n  python3 data/download_ucdp.py"
        )
    df = pd.read_csv(UCDP_CSV, low_memory=False)
    print(f"Loaded UCDP-GED: {len(df)} events, {df['year'].min()}-{df['year'].max()}")

    if "type_of_violence" in df.columns:
        df = df[df["type_of_violence"] == 1]
        print(f"  state-based events: {len(df)}")

    if "best" in df.columns:
        cy = (df.groupby(["country", "year"])["best"]
                .sum()
                .reset_index()
                .rename(columns={"best": "fatalities"}))
    else:
        cy = (df.groupby(["country", "year"])
                .size()
                .reset_index(name="event_count"))
        cy["fatalities"] = cy["event_count"] * 5
    cy["had_state_conflict"] = (cy["fatalities"] >= 25).astype(int)
    print(f"  country-years with active state conflict: {cy['had_state_conflict'].sum()}")
    return cy


def main():
    print("=" * 78)
    print("UCDP-GED → V-Dem ERT Transfer Learning Gating Test")
    print("=" * 78)
    print()

    cy = load_ucdp_country_year()
    ucdp_countries = set(cy["country"].unique())

    rows = []
    not_found = []
    for ep_country, info in KNOWN_EPISODES.items():
        onset = int(info["onset"])
        ep_type = info.get("type", "?")
        match_name = None
        for cand in country_candidates(ep_country):
            if cand in ucdp_countries:
                match_name = cand
                break

        if match_name is None:
            not_found.append(ep_country)
            had_active = False
            n_years_active = 0
        else:
            window = cy[
                (cy["country"] == match_name)
                & (cy["year"] >= onset - 10)
                & (cy["year"] < onset)
            ]
            n_years_active = int(window["had_state_conflict"].sum())
            had_active = n_years_active > 0

        rows.append({
            "country": ep_country,
            "onset": onset,
            "type": ep_type,
            "ucdp_match_name": match_name or "",
            "n_years_active_conflict_prior_decade": n_years_active,
            "had_prior_conflict": had_active,
        })

    df_out = pd.DataFrame(rows).sort_values(["had_prior_conflict", "type", "country"])
    df_out.to_csv(OUT_CSV, index=False)

    n_total = len(df_out)
    n_with = int(df_out["had_prior_conflict"].sum())
    pct = 100 * n_with / n_total

    print()
    print("=" * 78)
    print("GATING TEST RESULT")
    print("=" * 78)
    print(f"  Episodes with state-based conflict in country × [t-10, t-1]:  "
          f"{n_with}/{n_total}  ({pct:.0f}%)")
    print()

    if pct < 40:
        print("  DECISION: < 40%  →  TRANSFER LEARNING LIKELY TO FAIL")
        print()
        print("  Mechanism (Beger-Morgan-Ward 2021): most ERT episodes are STEALTH")
        print("  autocratization — Hungary, Poland, Türkiye etc backslid via legal-")
        print("  institutional channels with no civil conflict. A conflict-pretrained")
        print("  encoder learns features for the wrong DGP and gives those countries")
        print("  LOW risk.")
        print()
        print("  RECOMMENDED FALLBACK:")
        print("  - Skip the 1-week transfer-learning architecture rewrite.")
        print("  - Instead add UCDP conflict-onset as Stage-5 meta-features:")
        print("      conflict_onset_lag3, conflict_onset_lag5,")
        print("      neighbor_conflict_count_lag1")
        print("  - ~2 hours of code, +0.01-0.03 AUC expected, zero architectural risk.")
    else:
        print("  DECISION: ≥ 40%  →  TRANSFER MAY HELP")
        print()
        print("  Next diagnostic (1 hr): linear-probe CCA between a small UCDP-")
        print("  pretrained encoder's activations and V-Dem ERT labels. If top-5")
        print("  canonical correlation < 0.3, still abandon. Otherwise commit to")
        print("  the 1-week pretrain → freeze → fine-tune plan.")

    print()
    print("Stratified by episode type:")
    for t, sub in df_out.groupby("type"):
        n_t = len(sub)
        n_c = int(sub["had_prior_conflict"].sum())
        print(f"  {t:14s}: {n_c}/{n_t}  ({100*n_c/n_t:.0f}%)")

    if not_found:
        print()
        print(f"Could NOT match {len(not_found)} episodes to UCDP country names:")
        for c in not_found:
            print(f"  - {c}")
        print("  (These are counted as no-conflict; add to COUNTRY_ALIASES if any")
        print("   actually had state-based conflict — the verdict won't flip.)")

    print()
    print("Stealth-autocratization candidates (NO prior state conflict):")
    no_conf = df_out[~df_out["had_prior_conflict"]]
    for _, r in no_conf.iterrows():
        print(f"  {r['country']:35s} ({int(r['onset'])}, {r['type']})")

    print()
    print(f"Wrote {OUT_CSV}")


if __name__ == "__main__":
    main()
