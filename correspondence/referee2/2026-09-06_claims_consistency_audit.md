# Claims-and-consistency audit of v28 (2026-09-06, evening)

Scope: every quantitative claim in the abstract, introduction and discussion traced to its statement in results or appendices; every qualitative assertion made in more than one place checked for contradiction; every table and figure reference checked against the float; plus a grep for every value superseded by the day's retrain. Three parallel readers covered the three sweeps; the fixes below are applied and the auditor now runs 249 checks, 249 PASS (from 238).

## Findings and fixes

Cross-section copy errors (all fixed):
- Intro said the United States ranks twelfth of 135; results say fifteenth.
- Intro and conclusion quoted the gradient booster's strict AUC as 0.674; Table strict and Section 5.1 say 0.667. The 0.674 is the ERT shared-representation figure from another sentence.
- Limitations and conclusion rounded the framework's strict AUC to 0.680; the table says 0.681.
- Limitations called the closure effect -0.223, which is Hungary's seed-mean direction; the factorial gives -0.231.
- Section 5.1 called the ERT framework 0.735; the table says 0.733. Section 5.1 also said "31 onsets" where the 933-row table rests on 28.
- Section 5.7 gave the lead-time decay as 0.956 to 0.835; the appendix, its figure and the CSV all say 0.949 to 0.858.
- The ROC/PR caption placed the precision peak at 0.78 at 30 percent recall; the plotted curve peaks at 0.77 at 25 percent.
- The permutation-importance paragraph called v2cademmob and v2smgovdom adjacent; they rank third and sixth.
- A hard-coded "Appendix E" pointed at Detection Detail instead of Factor and State Counts; a Section 4 pointer for the winsorization discussion now points at Appendix B.

Stale pre-retrain numbers the auditor's presence-only matching could not see (all regenerated from the corrected pipeline's outputs):
- Baseline table framework row 0.939 / 0.591 (now 0.938 / 0.622) and the sentence beside it.
- Table dsp_ablation (0.905 / 0.899 / 0.774 / -0.006, now 0.887 / 0.897 / 0.776 / +0.010); the sign of the DSP-removal delta had been wrong, so the "reproduces to within 0.002" reconciliation with Table channel_ablation is now exactly true.
- Appendix B channel weights (0.27 contiguity-dominant, now 0.28 cultural-dominant) plus a leftover "contiguity marginally dominant" sentence.
- Section 5.6 locked cross-section list (Serbia 0.468 / Hungary 0.455 / Ukraine 0.319 / Poland 0.105, now 0.462 / 0.415 / US 0.291 / Ukraine 0.281 / Poland 0.126) and "Serbia reads 0.468 here and 0.471 there" (now 0.462 / 0.465).
- Appendix A stratified LOEO table (warning 22/46, alert 12/46, now 18/46 and 11/46, matching the performance table) and its prose; the auditor's own text anchors for C4 had been pointing at the stale strings.
- Appendix J prose sign counts (Turkiye positive in all ten, Tunisia and Ukraine in nine).

Diagnostics that quoted pre-retrain values and were never in the cascade, now re-derived:
- Elastic-net pruning (Appendix G): 80 of 212 features selected (was 89); hold-out AUC 0.940 pruned against 0.938 (was 0.932 / 0.934); AUC-PR 0.622 to 0.579 (was 0.656 to 0.602). Coefficient table rebuilt from the seeded run's file. Auditor C16 re-anchored to the pruned row.
- Winsorization mode (Appendix B): two full refits, symmetric versus upper-tail. Kappa 0.717 under both; hold-out AUC 0.940 against 0.935; AP 0.633 against 0.584; LOEO watch 36 against 38 of 46; 296 of 29,576 beta cells differ (the old sentence's "90 of 7,394 observations" mixed cells and rows). "Leaves the downstream results intact" is withdrawn in favour of the measured movement. New auditor C41.

Overclaims and unbacked figures:
- "Indistinguishable to three decimals under either outcome" held only for ERT (our set differs by 0.009); rewritten.
- The contrast MDE range "0.06 to 0.08" omitted the 0.053 minimum; now 0.05 to 0.08.
- The intro's "base rate of roughly eight percent throughout" is false across designs (0.073 to 0.204); rewritten.
- The 44 percent of country-years signed toward autocracy was stated only in Section 8; now stated in Section 5.6 with its N (4,940) and covered by C38.
- "Approximately 274 positive country-years" is exact: 274 of 3,383 with post-onset years excluded; now stated so and covered by new C40.
- South Korea's 2025 polyarchy was given as 0.722; V-Dem v16 says 0.819 (Portugal's 0.899 to 0.822 was right).
- The baseline table's caption now explains why its percentile intervals differ from the performance table's BCa intervals on the same 736 rows.

Checked and consistent: the sixfold gap is everywhere a one-run artifact; the framework-versus-baselines comparison is everywhere unresolved; BSS language is consistent; LOEO and fifteen-episode refit figures agree everywhere; registration dates are everywhere inside the window; the size-matched arm is everywhere "remains after matching"; the design-audit scope is everywhere three or four systems; the prospective table's tier counts and named countries match Section 7; factorial values match Sections 1, 5.2 and 8; the watchlist, cross-validation, temporal CV, episode, factor-count and network-robustness tables match their text; no dangling references.

## Reproducibility note

A full-pipeline refit in a clean worktree reproduces the canonical run to within 0.002 of hold-out AUC (0.940 against 0.938) and one LOEO episode (36 against 35 of 46). Stages 4 and 5 are seeded; the residual is under investigation (Stage 3 initialisation or thread-level nondeterminism). All within-experiment comparisons in the paper (winsorization, elastic-net pruning, LOEO refits) run both arms as refits, so this tolerance does not touch them.

## Left for the author
- The GDELT row in Table data_sources gives 1990 to 2025 (the years used) where Section 3.3 gives the archive as 1979 to present; a column note would settle it.
