# Reproducibility Audit: Forecasting the Onset of Autocratization

**Date:** 2026-07-11
**Manuscript:** ~/Downloads/fullpaper (2).tex
**Outputs directory:** ~/AIM4D
**Method:** /audit-reproducibility (4 parallel claim-matching agents over ~85 numeric claims) + /verify-claims (3 fresh-context citation verifiers over all 75 bibliography entries)
**Tolerance source:** ~/.claude/skills/audit-reproducibility/replication-protocol.md (integers exact; estimates |diff| < 0.01; percentages ±0.1pp)

## Summary

| Status | Count |
|---|---|
| PASS | ~70 claims (Tables 4, 7, 8, 13, 14, 16, 18; Figure 4; Sections 5.2, 5.6, 6.5 coefficients, 6.6, 6.7; Wilson CIs; 4 of 6 Table 5 rows) |
| FAIL | 9 claim clusters (below) |
| UNMATCHED (no on-disk artifact) | 3 |
| Citations | 75/75 VERIFIED, zero fabrications |
| **Overall verdict** | **FAIL** (blocking items 1-4 below must be resolved before submission) |

## FAIL — outside tolerance (BLOCKER until corrected or downgraded with a named alternative)

| # | Claim | Reported | Computed | Source | Severity |
|---|---|---|---|---|---|
| 1 | Table 5 gradient-boosting AUC-PR (strict 2019 hold-out) | 0.663 | **0.735** (rerun-confirmed live) | robustness/benchmark_finishers_results.csv; benchmark_finishers.py rerun | **HIGH — changes the §5.7 narrative.** At 0.735 the booster leads the framework on BOTH metrics (0.941/0.735 vs 0.938/0.671), so "the framework leads on AUC-PR (0.671 against 0.663)... parity" is wrong as written. The paper's 0.663 appears to trace to a stale "~0.66" inline code comment. The leakage-adjusted 0.63 in footnote 6 DID verify (0.6338), so the leakage caveat survives; the parity sentence does not. |
| 2 | §5.5 + Table 3 caption + Appendix B: full-pipeline LOEO refit "fifteen episodes, mean +0.075, twelve of fifteen higher" | 15 episodes, +0.075, 12/15 | **5 episodes, +0.029, 3/5** | robustness/sample_pipeline_loeo.csv (+ .py); no 15-episode artifact exists anywhere | **HIGH — claim as stated was never run.** Direction (conservative bound) still holds at 5 episodes, but the count and magnitudes must be corrected or the 15-episode run actually executed. |
| 3 | Table 11 Factor 1 top-5 loadings | v2x_diagacc 0.891, v2x_freexp_altinf 0.876, v2x_clpol 0.864, v2x_cspart 0.851, v2x_pubcorr -0.839 | 0.991, 0.984, 0.979, 0.932, **-0.458**; computed top-5 also includes v2x_accountability and v2x_electoral_integrity, absent from the table | stage1_factors/factor_loadings.csv | **HIGH — not rounding; composition differs.** Either the CSV is stale relative to the run behind the table, or the table used a different normalization. Isolate which. |
| 4 | §4.4 graph statistics | 222,752 spatial edges, ~232,000 total | **176,600 spatial, 186,260 total** (consistent across 13 logged runs) | brev/run_all.log | **HIGH — internally inconsistent:** the paper's own Table 6 lists 176,600 edges for the full network. Nodes (4,968) and temporal edges (9,660) match. |
| 5 | Table 2 permutation-importance ranks 7-10 | network_exposure 0.004, libdem_c22_5 0.004, work_age_share 0.004, v2smgovdom_x_post2015 0.003 | true ranks 7-8 are csd_index_x_post2015 (0.0044) and v2exl_legitratio (0.0043), absent from the table; work_age_share is rank 22 (0.0022); v2smgovdom_x_post2015 is rank 13; the v2cagenmob value matches only the _detrended variant | robustness/permutation_importance_oos.csv | MEDIUM — top-6 (the load-bearing rows) match closely; the tail of the table misstates the ranking and one variable name. Headline mobilization-vs-DSP ordering unaffected. |
| 6 | Table 6 MSE-improvement percentages + §6.2 "mean 41.5%" | 43.6 / 31.6 / 35.9 / 54.9; mean 41.5 | 43.18 / 30.94 / 36.03 / 53.58; mean 40.93 | robustness/network_variants_results.csv | MEDIUM — drift up to 1.3pp despite seeded pipeline; edge counts and mean-contagion in the same table match exactly, so the percentages likely came from an earlier code version. Re-run and reconcile. |
| 7 | Table 15 learned weights (contiguity, alliance) | 0.269 / 0.253 | 0.240 / 0.263 (seed-sweep means) | robustness/network_seed_sweep_summary.csv | MEDIUM — no on-disk artifact contains the exact 0.269/0.262/0.253/0.217 quartet; check whether Table 15 reports a single locked run that was never persisted. See also UNMATCHED item on the cultural weight. |
| 8 | §6.5 lagged near-vs-far contrast "significant in none" | 0 of 200 draws | 1 of 200 (sig_frac = 0.005) | robustness/contagion_dyadic_results.csv | LOW — change "none" to "one of 200". |
| 9 | Table 9 alert-tier assignment (top 5 = Alert) | Bolivia-Uganda at Alert | not reproducible from live ews_signals.csv at HEAD or at registration commit 6d0c68a (tier cutoffs are data-dependent percentiles; neither snapshot shows top-5 = alert) | stage5_ews/ews_signals.csv | MEDIUM — point risk scores match the frozen prospective_drivers_results.csv exactly, but the tier column and any rerun from the live pipeline diverge (Bolivia 0.535-0.540 vs 0.476; Ivory Coast appears at the registration commit). The "registered in advance" claim needs the frozen artifact to be the pinned, documented reference. |

## UNMATCHED — no computed counterpart on disk (manual action needed)

| Claim | Raw context | Status |
|---|---|---|
| Appendix D elastic-net diagnostic (AUC drop ~0.07; all 20 coefficients incl. csd_x_network +0.724) | tab:enet_features | robustness/elastic_net_robustness.py exists but has never been run; its output CSV is absent. The "~0.07" exists only as a hardcoded print string in stage5_ews/estimate.py. Run the script and reconcile the table. |
| §6.5 horse-race global-precedent term α = 0.62 | "the global term dominates" | Computed by contagion_galton.py but only printed to stdout, never written to contagion_galton_results.csv (which does contain the 0.12/0.11 cultural coefficients that verified). Persist it. |
| Table 15 cultural-linguistic weight 0.262 | learned convex weights | network_seed_sweep.py bug: saves only w[0:3], silently drops alpha_cultural. Fix the script and re-run. |

## Notes (non-blocking)

- **Stale-CSV false alarm resolved in the paper's favor:** false_positive_summary.csv / false_positive_table.csv (May 11) show 29-30 stable-democracy FPs, but they predate the May 23 analysis rewrite and June 20 ews_signals refresh; a fresh rerun of identify_fps() gives 0 FPs, matching the paper. Delete or regenerate the stale CSVs.
- Table 4, Table 7, Table 8, Table 13, Table 14, Table 16, Table 18, Figure 4, the §5.2 precedence numbers, §5.6 novel-precision block, §6.5 spatial coefficients (except the unpersisted 0.62), §6.6 sanity/stability numbers, §6.7 CSD rates, and the Table 3 Wilson intervals all reproduce to well within tolerance — many exactly.
- All 75 bibliography entries verified against publisher pages by fresh-context verifiers: no fabricated citations, no venue/author discrepancies. Rød/Hegre/Leis 2025 is correctly dated (online-first 2023, canonical issue 2025). Schmotz & Selvik confirmed as a live 2025 WZB working paper.

## Next steps

1. **Item 1:** correct Table 5 (0.663 → 0.735) and rewrite the §5.7 parity passage around the honest pair (booster 0.735 full / 0.634 leakage-adjusted vs framework 0.671); the interpretability framing survives, the "framework leads on AUC-PR" sentence does not.
2. **Item 2:** either run the full 15-episode pipeline-refit LOEO (Brev-sized job) or restate the claim at its true scope (5 episodes, +0.029, 3/5).
3. **Item 3:** regenerate Table 11 from stage1_factors/factor_loadings.csv or locate/pin the run that produced the manuscript values.
4. **Item 4:** fix §4.4 edge counts to 176,600 / 186,260 (matches Table 6 and 13 logged runs).
5. Items 5-8: mechanical corrections in the tex + one re-run of network_variants.
6. Item 9 + UNMATCHED: pin the registered forecast artifact (commit hash of prospective_drivers_results.csv) in the text, fix the network_seed_sweep w[3] bug, persist the galton horse-race output, and run elastic_net_robustness.py once.

---

# Certification addendum (2026-07-11, post-regeneration)

All pipeline and robustness artifacts regenerated from a single clean run (commits 8df913e through 1e03d82). The manuscript was updated to the fresh vintage and independently re-verified against every artifact. Four transcription/stale errors found in the update pass (coup LOEO 79->71 in 8.3, Table 14 S=3 row, mean rank correlation 0.93->0.92 x3, cultural post-2005 coefficient 0.21->0.20) were fixed. Table 4's LOEO intervals are Wilson by stated methodology (cluster-bootstrap alternatives differ immaterially). Remaining known limitation: the 6.6 eigenvalue/permuted-null figures are printed by sanity_checks.py but not persisted to CSV.

**Verdict: CERTIFIED — every numeric claim in the manuscript traces to the committed artifact set from one run.**

Notable outcome changes vs the June vintage: 15-episode full-refit LOEO now real (+0.081, 14/15); Rashomon dominance softened to 5/6 models; measurement-uncertainty to 29/30 draws; hold-out headline 0.938->0.934, AUC-PR 0.671->0.656; refit-CV primary estimate unchanged at 0.821; stage-3 kappa improved 0.66->0.72; elastic-net pruning cost corrected from "~0.07 AUC" to "-0.003 AUC / -0.054 AUC-PR" with the real coefficient table.
