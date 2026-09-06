# Referee 2 Report, Round 1

**Project:** YCRG-Labs/AIM4D, manuscript "Forecasting Autocratization: Predictive Performance and Sensitivity to Evaluation Design" (working copy `~/aim4d-paper/fullpaper.tex`, v25 at audit start)
**Date:** 2026-09-06
**Protocol:** Cunningham's Referee 2 (five audits). Audits 1a, 1b, 2, and 3+4 were run by independent Sonnet subagents with read/run-only access to the author's tree; Audit 5, the paper review, and the size-matched closure test were run by the main thread. Replication scripts are in `code/replication/`. No author file was modified by the audit; fixes applied to the manuscript in response are listed at the end and were made by the author-side editor, not the audit.

## Summary

The strict-design pipeline's mechanical plumbing is sound: rolling-origin closure lags are implemented consistently across the ledger and ERT scripts, `EXCLUDE_COLS` is verifiably complete against every label-derived column in `ews_signals.csv`, merges lose no rows, scalers fit on training folds only, and every headline number in Tables 5 and 6 reproduces in R to three decimals from the committed CSVs. Reproducibility of the committed summary tables is byte-exact. The one numeric discrepancy found (Table 6 closure marginal, -0.224 reported versus -0.2235 in the current CSV) is a stale digit, not a coding error, and has been corrected. The paper's central mechanism claim survives its own falsification test once a size-matched arm is added: roughly four-fifths of the closure effect is label memorization across the window boundary and one-fifth is training-set size.

The serious findings are elsewhere. Stage 4 has two leaks and one bug that reach flagship claims: bidirectional temporal edges contaminate the training loss, not just decoding; the `EXCLUDE_COUNTRY` guard is defined but never used in Stage 4, so the fifteen-episode full-pipeline LOEO refit is leave-one-out at four stages and not five; and the econ-similarity edge set is computed on `abs()` of a z-scored income series, which makes rich and poor countries "similar." The SDEM spillover intervals in Appendix C.7 are generated from a theta=0 process and are therefore centered on zero by construction. On the replication package: all 21 tables and every pgfplots figure are hand-typed into the LaTeX with no generation path; the numeric-claim auditor covered roughly a quarter of the manuscript's figures and none of the headline ones at audit start (now 200 checks, all headline figures covered); and the auditor's own data dependency, `quality_reports/`, was gitignored, so the public repo could not run it (now tracked).

Verdict: **Major revisions.** The design finding is real, well falsified, and portable. Stage 4 needs three code fixes and a retrain before any network-decomposition claim or the LOEO refit can be cited without caveat, and the output pipeline needs to be made reproducible.

---

## Audit 1a: Code Audit (forecasting-claim scripts)

Scope: `onset_forecast_clean.py`, `strict_table_final.py`, `design_factorial.py`, `closure_placebo.py`, `ert_panel.py`, `design_factorial_ert.py`, `onset_forecast_ert.py`, `strict_ert_sensitivity.py`, `episode_ledger.py`.

### Findings

1. **[HIGH, resolved by follow-up test]** `closure_placebo.py` — The placebo shows the closure gap *grows* under permuted onsets (12/12 permutations more negative than the observed -0.190; placebo mean -0.313, sd 0.036). The subagent read this as falsifying a "leakage" interpretation. The manuscript does not claim foresight-leakage; it claims label memorization, which a growing placebo gap predicts. But the subagent's underlying point stands: the placebo cannot distinguish memorization from the closed arm simply training on four fewer years of rows, since both survive permutation. **Follow-up:** `code/replication/referee2_closure_sizematch.py` adds a third arm, the open training set subsampled at each origin to the closed arm's row count, identical seeds in all arms, 3 learners x 3 seeds. Result: closed - open = -0.190 (reproduces the placebo script's observed exactly); closed - open_matched = -0.150; open_matched - open = -0.040. **79% of the closure effect is memorization, 21% is sample size.** The mechanism claim is supported and is now stated in this form in the manuscript (Sections 5.2 and 8.1).

2. **[MEDIUM]** `closure_placebo.py:102` vs `:118` — The observed statistic averages 3 learners x 2 seeds; each placebo draw averages 3 learners x 1 seed. Asymmetric averaging inflates the "N placebo SDs" statistic at line 131. Fix: identical seed sets in both. (The size-matched test above uses identical seeds in every arm.)

3. **[MEDIUM]** `closure_placebo.py:41,132` — `N_PERM=12`, share printed to three decimals. Resolution is 1/12. Fix: raise N_PERM by an order of magnitude or report k/n.

4. **[MEDIUM, dormant]** `strict_table_final.py:154-164`, `strict_ert_sensitivity.py:127-137` — Rank-mean blend fills a failed component's predictions with that component's *pooled* median across all origins, a future-information channel if any fit ever fails. Verified not live: all six models have byte-identical row sets across 13 origins, so no fit failed in the run behind Table 5. Fix: per-origin fallback.

5. **[LOW/MEDIUM]** `onset_forecast_ert.py:88-119` — Dead copies of `build_panel()`/`label_h()` from the ledger script are never called but import `KNOWN_EPISODES`, misleading in a file whose purpose is to avoid the ledger. Fix: delete; import shared pieces.

6. **[LOW]** `onset_forecast_clean.py:95-105` — Ledger episode membership ends at `peak`; ERT's ends at `end`. Five country-years (Poland 2020-2021, Slovenia 2025, Greece 2025, Guyana one year) re-enter the at-risk pool while ERT codes the episode ongoing, 0.3% of 1,834 at-risk rows. Now disclosed in Section 5.1.

7. **[LOW]** `strict_ert_sensitivity.py:76-89` — Panel B has no explicit common-row intersection step (Panel A does). Verified aligned (n=1002) but by an implicit invariant. Fix: add the intersection.

8. **[LOW]** `design_factorial.py:38,64-68` — Fixed-2019 arm scores calendar year 2020 only (n=70, 10 positives). Already disclosed in Section 5.2.

9. **[LOW]** `fillna(0)` on raw features at fit and score time across all strict scripts, no missingness indicator. 0.4% of cells, concentrated in lag/delta columns (structural first-year missingness). Not leaking; undocumented. Fix: note in data appendix.

10. **[INFO]** `strict_table_final.py:40` — Module-level RNG state carries across `model_ci` calls into `paired_ci`; deterministic but fragile to refactoring. Skip rule `y.sum()<3` fires 0/2000 (verified).

### Checks run
Leakage-column scan of `ews_signals.csv` (14 label-derived columns, all in `EXCLUDE_COLS`); row-set identity across all six Panel A models (n=933, origins 2008-2020); Panel B n=1002, origins 2005-2020; merge diagnostics on `build_panel()` (3580 -> 3580, 0 duplicates); `ert_panel` at-risk = 1621; pandas 3.0.5 `attrs` propagation verified; `closure_contrast()` rerun reproduces -0.1902; factorial closure marginal recomputed at 0.2235 pooled / 0.1898 in the strict corner.

---

## Audit 1b: Code Audit (pipeline stages, end-to-end, spatial)

Scope: `stage1_factors/extract.py`, `stage2_betas/estimate.py`, `stage3_msvar/estimate.py`, `stage4_nscm/estimate.py`, `stage5_ews/estimate.py`, `robustness/strict_endtoend_refit.py`, `robustness/_refit_worktree.py`, `robustness/spatial_models.py`, `robustness/spatial_gm.py`, `data/compute_changepoints.py`, `robustness/sample_pipeline_loeo.py`.

### Findings

1. **[HIGH]** `stage4_nscm/estimate.py:179-186` — Temporal edges added in both directions; 2-layer `GCNConv` runs over train+test nodes with no edge masking, so post-cutoff nodes shape training-period representations and the training loss `F.mse_loss(y_full[mask_train], ...)`. This is the Stage 3 forward-backward leak's analogue, but it contaminates gradient-based training. The end-to-end refit truncates inputs and so bounds this for the forecasting comparison; it does not bound it for the network decomposition (Section 5.6, Appendix J). Fix: past-to-present edges only, or causal sub-graph masking per node-year. Now disclosed in Section 8.3.

2. **[HIGH]** `stage4_nscm/estimate.py:23` + `robustness/sample_pipeline_loeo.py:65` — `AIM4D_EXCLUDE_COUNTRY` is defined in Stage 4 and never referenced (grep-verified); Stages 1, 3, 5 gate on it correctly. The fifteen-episode full-pipeline LOEO refit (passport C5, "+0.081, fourteen of fifteen higher") therefore trains the shared GCN encoder on the held-out country's pre-cutoff trajectory. The `resid_nscm_*` features Stage 5 consumes for that country come from a model that has seen it. Fix: propagate the guard into `mask_train`. Now disclosed in Section 8.3 as a bound on the refit.

3. **[HIGH]** `robustness/spatial_models.py:268-284` vs `328-333` — The file's docstring says each family is bootstrapped from its own fit; true for SAR/SAC, false for SDEM. `SDEM.lambda` and the reported `theta_W.{c}` intervals are generated from `y_star = Xe @ beta_e + neumann(e, lam_e)`, a pure-SEM DGP with no `WX`/theta term. The theta intervals therefore describe noise around a true theta of zero and are centered there by construction; "individually indistinguishable from zero" in Appendix C.7 is not evidence. Fix: regenerate from `base["SDEM"]`'s own beta/theta/lambda. Now caveated in C.7.

4. **[MEDIUM]** `stage4_nscm/estimate.py:517-523` vs `608` — `network_ablation_test` receives the un-split `mask_test` (selection + eval rows), so `improvement_spatial_edges` / `improvement_total_network` are partly scored on checkpoint-selection rows. Fix: pass `mask_eval`.

5. **[MEDIUM]** `stage4_nscm/estimate.py:169-177` — Econ-similarity edges: `gdp_vals = node_features[..., -2]` is the *z-scored* `gdp_pc`; `log_gdp = np.log1p(np.abs(gdp_vals))`. A country at z=+2 and one at z=-2 register as similar. The channel does not measure economic similarity. `alpha_econ = 0.22` is the weight of a mis-specified edge set. Fix: compute on raw GDP per capita. Now disclosed in Section 3.2.

6. **[MEDIUM, confirmed by data]** ATOP 5.1 ends 2018. With the 5-year trailing window, active alliance dyads: 25,053 (2018) -> 20,932 -> 16,810 -> 12,679 -> 8,454 -> 4,228 (2023) -> **0 (2024, 2025)**. The alliance channel is dead for the prospective origin and decayed across the whole evaluation window, with no diagnostic printed. Already disclosed in Section 3.2 (added earlier this round). Fix: per-year coverage diagnostic; freeze the network at last observed state or drop the channel post-2018.

7. **[MEDIUM, confirmed]** `stage3_msvar/estimate.py:313-348` — `hamilton_filter_fast` runs a full forward-backward pass on the untruncated sequence (`decode_all` uses `X_all`), so state posteriors at t condition on t+1..T, including for TVTP covariate selection at lines 624-625. The manuscript admits this (Section 5.3) and the end-to-end refit exists to bound it. Consistent with the 12-14 point gap between hold-out (C2) and refit CV (C1) in the passport.

8. **[LOW-MEDIUM]** `spatial_models.py:175-183` via `spatial_gm.py` — Lee-instrument fixed point iterates to 40 with tol 1e-6 but sets no convergence flag; non-converged `rho0` used silently inside ~600 bootstrap draws. Fix: track and filter on convergence.

9. **[LOW]** `stage1_factors/extract.py:36-51` — Indicator eligibility (the 20% missingness filter) is computed over the full year range, so post-cutoff missingness can determine the pre-cutoff feature set in the non-refit pipeline. Neutralized by truncation in the refit.

10. **[CLEAN]** Stage 2 Kalman smoother is genuinely one-sided post-cutoff (hyperparameters on `y_train`/`x_train`, RTS over the pre-cutoff subset only, post-cutoff betas held at the last smoothed value).

11. **[CLEAN]** Seeds: Stages 1-2 use no RNG (deterministic eigh/svd/L-BFGS-B); Stage 4 seeds torch and numpy; seed-sweep scripts vary seed deliberately. Cross-machine bit-identity not guaranteed for near-degenerate eigenpairs.

12. **[CLEAN, fragile]** `data/compute_changepoints.py:42-66` — PELT window strictly trailing `[t-29, t]`; row-filtering `changepoints.csv` by year is equivalent to per-origin recompute. Fragile: `compute_changepoints.py` is not in `_refit_worktree.STAGES`, so any future non-trailing change would silently break truncation.

13. **[CLEAN, open item]** `truncated_data_dir` recurses correctly (verified the ATOP-symlinked-whole bug is fixed) and truncates every year-indexed input checked. Not verified: `catch22_features.csv`, `global_diffusion.csv`, `pitf_regime.csv`, and GDELT/Archigos/UCDP derivatives are not in `STAGES`; each must be independently trailing-window by construction. Spot check of `catch22_features.csv` is consistent with that but not proof.

14. **[INFO]** Leakage surface of `ews_signals.csv` enumerated: 14 columns, all in `strict_endtoend_refit.EXCLUDE`. ~40 other robustness scripts read this file and were not individually re-verified.

---

## Audit 2: Cross-Language Replication (R)

### Replication Scripts Created
- `code/replication/referee2_replicate_A.R` — Table 5 Panel A, six-model AUC/AP, structural checks
- `code/replication/referee2_replicate_B.R` — paired country-clustered bootstrap, 5 seeds x 2000
- `code/replication/referee2_replicate_C.R` — Table 6 marginals, both outcomes, corners
- `code/replication/referee2_closure_sizematch.py` — size-matched closure arm (main thread, see Audit 1a item 1)

AUC (Mann-Whitney, mid-rank ties) and AP (sklearn's grouped-threshold definition) implemented by hand; `quantile(type=7)` matches numpy's default.

### Comparison Table

| Quantity | Python (author) | R (referee) | Match? |
|---|---|---|---|
| Panel A rows / positives / origins | 933 / 107 / 2008-2020 | 933 / 107 / 2008-2020 | Yes |
| Persistence AUC / AP | 0.591 / 0.210 | 0.591 / 0.210 | Yes |
| Four polyarchy AUC / AP | 0.718 / 0.226 | 0.718 / 0.226 | Yes |
| Elastic net AUC / AP | 0.654 / 0.194 | 0.654 / 0.194 | Yes |
| Gradient boosting AUC / AP | 0.674 / 0.220 | 0.674 / 0.220 | Yes |
| Random forest AUC / AP | 0.715 / 0.205 | 0.715 / 0.205 | Yes |
| Framework blend AUC / AP | 0.680 / 0.203 | 0.680 / 0.203 | Yes |
| Paired dAUC point | -0.038 | -0.0379 to -0.0403 (5 seeds); 50k-draw: -0.0382 | Yes, MC error |
| Paired dAUC CI (fresh seed) | [-0.113, +0.040] | [-0.115, +0.039] | Yes, within 0.005 |
| Factorial (ledger) risk / label / origin | +0.031 / -0.013 / -0.021 | +0.031 / -0.013 / -0.021 | Yes |
| Factorial (ledger) closure | **-0.224** | **-0.2235** | **Off by 0.0015: stale digit** |
| Factorial (ledger) ranges, all four | as reported | exact | Yes |
| Corners (ledger) | 0.874 / 0.662 | 0.874 / 0.662 | Yes |
| Factorial (ERT) all four + ranges | +0.018 / -0.002 / -0.088 / -0.138 | exact | Yes |
| Corners (ERT) | 0.883 / 0.664 | 0.883 / 0.664 | Yes |

### Discrepancies Diagnosed
One. The ledger closure marginal recomputes to -0.223483 from the current `design_factorial.csv`; the manuscript carried -0.224 from before the file was regenerated at commit `9b653e8`. Every individual pairwise difference and both range endpoints match exactly, so it is a manuscript-freshness issue, not a computation error. **Corrected in the manuscript (six occurrences) and now covered by auditor check C29.**

### Verdict
No evidence of a Python computation bug in AUC, AP, bootstrap, or factorial pairing.

---

## Audit 3: Directory & Replication Package

### Replication Readiness Score: 4/10

### Deficiencies
1. No raw/clean/code/output separation; every stage directory and `robustness/` interleave scripts with their CSV outputs (118 `.py` and ~200 CSVs in one flat directory).
2. Three non-overlapping "master" scripts (`run_all_local.sh`, `run_phase2_local.sh`, `robustness/run_all.py`) together invoke 37 of 118 robustness scripts. 81 scripts have no documented place in any execution order.
3. README documents only the 7-command canonical pipeline; none of the master scripts, the auditor, or the reason `prospective_drivers.py` is excluded from phase 2.
4. Data acquisition vague: no URLs for COW/ATOP; GDELT provenance ambiguous between `download_gdelt.py` and `build_gdelt_proxy.py`.
5. Python-version mismatch (`python3.11` default in shell scripts; only environment is `.venv_ablation`, 3.14.7); `requirements.txt` is `>=` only, no lockfile; installed numpy 2.5.2 / pandas 3.0.5 are not pinned.
6. **[Fixed this round]** `quality_reports/` including the passport YAML was gitignored, so the public repo could not run `audit_claims.py`. Now tracked (7 files).
7. Minor: `brev/run_all.log` (400KB) committed.

### Evidence (positive)
Zero hard-coded `/Users/` or `/home/` paths (two overridable `~/` defaults). Public remote, no credentials tracked. Seeds correct for every genuinely stochastic component. Large raw data properly gitignored.

---

## Audit 4: Output Automation

### Tables: **Manual** — 21 tables, zero `\input`/`\include`, no generated `.tex` fragments anywhere, no `pgfplotstable`/`csvsimple`. **MAJOR.**
### Figures: **Manual** — 21 `\addplot coordinates` blocks, 8 tikzpictures; no script under the repo writes coordinate strings. **MAJOR in practice.**
### In-text statistics
At audit start: `audit_claims.py` 122/122 PASS, but the passport's 26 entries covered ~25% of the manuscript's ~478 distinct decimals and **none** of the headline figures (0.693, 0.662, 0.138, 0.883, 0.664, 0.041, 0.031, 0.344, 48 onsets, the end-to-end paired differences). **[Fixed this round]** Checks C27-C34 added: 200/200 PASS, every headline figure covered, closure marginal verified at -0.223.
### Reproducibility: **exact** for reported statistics. `strict_table_final.py` rerun reproduces `strict_table_final_h5.csv` byte-for-byte; the raw predictions file differs at the 16th-17th significant digit (RandomForest `n_jobs=-1` summation order), which never reaches the paper.

---

## Audit 5: Econometrics

See `audit5_and_paper_review.md` appended below.

---

## Major Concerns (must be addressed)

1. **Stage 4 code, three items** (1b.1, 1b.2, 1b.5): bidirectional temporal edges into the training loss; `EXCLUDE_COUNTRY` unused so the LOEO refit is not leave-one-out; econ-similarity edges on `abs(z)`. All three require a Stage 4 retrain to fix. Until then the network decomposition (Section 5.6, Appendix J), the learned edge weights, and the fifteen-episode refit figure carry the caveats now written into the manuscript.
2. **SDEM theta intervals** (1b.3): regenerate from the Durbin-error fit. Until then C.7's "supports nothing about contextual effects" rests on intervals centered at zero by construction.
3. **Output pipeline** (Audit 4): 21 hand-typed tables and every figure with no code path from CSV to LaTeX. The 200-check auditor mitigates the tables; it does not mitigate the figures at all. Generate at least Tables 5 and 6 and the ROC/PR/reliability figures from the CSVs.
4. **Replication package** (Audit 3): one master script that runs every script a table depends on, pinned dependencies, README that matches.

## Minor Concerns

- Interval method inconsistent across tables (BCa in Table 2, percentile in Tables 3 and 5) for the same quantity, lower bounds 0.075 apart, unreconciled (Audit 5.1).
- Five-positive training rule has no robustness check and determines the scored origin set (Audit 5.2).
- Strict "framework" is a rank-mean blend, conventional is a stacked meta-learner; report single-learner strict AUCs (Audit 5.3).
- `closure_placebo.py` seed asymmetry and N_PERM=12 (1a.2, 1a.3).
- Dead ledger code in `onset_forecast_ert.py` (1a.5).
- `network_ablation_test` scored on selection rows (1b.4).
- Lee fixed-point convergence unflagged (1b.8).
- `fillna(0)` undocumented (1a.9).
- GDELT coverage stated as 1990-2025 in Table 1 and "around 2000" in Section 3.3.
- In-sample rows in Table 2 will be quoted out of context.

## Questions for Authors

1. Will Stage 4 be retrained with causal edges, the exclusion guard, and corrected econ-similarity, and the network sections regenerated? If not, the decomposition should be moved to an appendix labelled as exploratory.
2. Is there a reason the framework is combined differently under the two designs beyond the fold-availability argument, and what are the single-learner strict AUCs?
3. Why five positives?

## Verdict
[ ] Accept  [ ] Minor Revisions  [x] Major Revisions  [ ] Reject

**Justification:** The evaluation-design finding is correct, replicates in R, survives permutation and size-matching, and holds under two outcome definitions; it is the paper and it is publishable. The network decomposition, the LOEO refit, and the spatial spillover intervals rest on code with three identified leaks or bugs and cannot be cited as they stand. The output pipeline is not reproducible from the repository in the protocol's sense.

## Recommendations (prioritized)

1. Retrain Stage 4 with past-to-present edges, the `EXCLUDE_COUNTRY` guard in `mask_train`, and econ-similarity on raw GDP; regenerate Sections 5.6, Appendix J, the edge weights, and the C5 refit figure.
2. Regenerate SDEM bootstrap from its own fit; update C.7.
3. Generate Tables 5, 6 and the main figures from CSVs via a script; wire it into a single master runner.
4. Pin dependencies (`pip freeze > requirements.lock`); reconcile the Python version; update README.
5. Add a three-positive robustness run for Table 5 Panel A.
6. Unify the interval method across Tables 2, 3, 5.
7. Fix the closure placebo's seed asymmetry and raise N_PERM.

## Fixes applied to the manuscript in this round (author-side, not by the audit)

- Table 6 closure marginal -0.224 -> -0.223 (six occurrences).
- Size-matched closure decomposition (79% memorization / 21% sample size) added to Sections 5.2 and 8.1.
- Disclosures added: Stage 4 LOEO gap (8.3), bidirectional edges reach the training loss (8.3), econ-similarity mis-specification (3.2), SDEM theta intervals centered by construction (C.7), five readmitted country-years under the peak rule (5.1).
- `quality_reports/` tracked; auditor extended to 200 checks covering every headline figure.
## Audit 5: Econometrics

### Identification Assessment

This is a forecasting paper and does not claim causal identification; the object that needs identification is the evaluation-design effect, and the factorial identifies it correctly. Sixteen cells, three learners, three seeds, each cell differing from its neighbours in exactly one dimension, marginals averaged over the twenty-four paired comparisons in which only that dimension moves. That is a clean design. Two of its limits are already disclosed and correctly so: the fixed-origin cells score a single year (138 or 70 rows) against 933 for the rolling arm, so the origin marginal is estimated on almost nothing; and the label dimension is degenerate inside the at-risk pool because a country in its onset year is in-episode by construction. The placebo (permuted onsets, twelve draws, gap grows from -0.190 to -0.313) is the right falsification test for the mechanism and it is run correctly; it establishes that the closure effect is label memorization across the window boundary and not look-ahead about the onset process.

The spatial ladder (Appendix C.7) is the one place a structural parameter is estimated. Year dummies are in the design matrix, so the global wave is absorbed. The Lee best instrument gives first-stage F = 78.9 on the full panel, but the era split I ran shows F collapsing to 12.9 post-2005 and the interval running to +7.03, outside the stationary region. The paper now reports this. The full-panel rho = 0.333 [0.159, 0.537] is therefore carried by the pre-2005 democratization wave and says nothing about the backsliding era; the manuscript's Section 8 correctly no longer draws a transmission conclusion from it.

### Specification Issues

1. **[MEDIUM] Interval methods differ across tables for the same quantity, unflagged.** The 2019 hold-out AUC of 0.939 carries a BCa cluster-bootstrap interval of [0.800, 0.975] in Table 2 and a country-clustered percentile interval of [0.875, 0.982] in Table 3. Same point estimate, lower bounds 0.075 apart, no sentence reconciling them. A referee will read this as two different analyses. Fix: one interval method for all discrimination rows, or a footnote stating why BCa and percentile disagree here (BCa corrects for the skew that 19 effective episodes induce; the percentile interval does not).

2. **[MEDIUM] The five-positive training rule is a tuning choice that determines the scored sample and has no robustness check.** It sets Panel A to origins 2008-2020 and Panel B to 2005-2020, hence 933 versus 1,002 rows and 28 versus 48 onsets. Nothing in the paper motivates five rather than three or ten, and a reader cannot tell whether the framework-versus-four-variables sign reversal between panels is an outcome effect or a sample-composition effect (the paper's own common-origins check, +0.025 [-0.025, +0.078], suggests outcome, but that check only removes the 2005-2007 origins, not the rule). Fix: rerun Panel A under a three-positive rule and report whether the ordering moves.

3. **[MEDIUM] The "framework" in Table 5 is not the framework in Table 3.** Conventional results use a stacked meta-learner over the stage outputs; strict results use a rank-mean blend because the rolling design leaves no held-out fold to fit a stacker at early origins. The paper discloses this, but it means the drop from 0.939 to 0.680 conflates a combiner change with the design change. The factorial (Table 6) is immune, because it holds three single learners fixed; the headline framework figure is not. Fix: report the strict-design AUC of each of the three single learners alongside the blend, so the reader can see the blend is not doing the work.

4. **[VERIFIED, no issue] Clustering.** Country-clustered bootstraps are the right level: overlapping five-year windows put every positive row from one onset in one cluster. 93 countries in the at-risk pool exceeds the conventional 50-cluster floor, and although only 28 (ledger) or 42 (ERT) carry a positive, the `y.sum() < 3` skip rule in `paired_ci` never binds: I re-ran the resampling with the author's seed and it discarded 0 of 2,000 replicates under both outcomes. The lower tail is not truncated.

5. **[LOW] DSP imputation touches training rows under the strict design.** Pre-2000 Digital Society Project values are filled with panel medians computed on the whole panel. The paper argues this "enters only the training-period representation and never the evaluation." True for the scored rows, but training rows dated before 2000 carry values computed from post-origin data. The exposure is small (origins start 2005/2008, DSP is densely coded from 2000) and the block adds nothing under the strict design anyway, so the direction of any bias is toward the null on the DSP block. Fix: impute with the median through the origin year, or drop pre-2000 training rows from the strict design and confirm nothing moves.

6. **[LOW] Magnitude plausibility.** A 0.22 AUC design effect is large but the placebo makes it plausible: with a six-year window and a 2019 fixed origin, an onset in 2021 puts positive rows in 2016-2020 (training) and 2021 (test) for the same country, so the learner can key on country identity. Nothing implausible here; if anything the effect should be expected in every window-labelled panel forecast.

7. **[LOW] In-sample rows in Table 2.** AUC 0.993 and 46/46 detection are in-sample and are labelled as such, but they sit in the paper's main performance table two rows below out-of-sample figures. A referee skimming the table will quote them. Fix: move in-sample rows to an appendix or a separate table.

### Power (Nyhan item 8)

Now stated correctly in the manuscript. From the paired intervals, MDE at 80 percent power: framework vs four variables 0.111 (strict, ledger) / 0.076 (strict, ERT) / 0.149 (end-to-end, ledger) / 0.086 (end-to-end, ERT); mobilization block 0.041 / 0.031; digital-control block 0.046 / 0.057; end-to-end vs shared representation 0.099 / 0.053. The consequence the paper draws is right: the strict null on mobilization was powered to find the conventional-design effect of 0.042, so it is informative; the model comparison could only have detected a difference the size of the design effect itself, so it is not.

---

## Paper Review (Humphreys structure; Edmans / Nyhan / Blattman / Evans applied)

### Part 1: Summary

The paper's contribution is a measurement of how much of an autocratization forecast's apparent accuracy is produced by the evaluation design rather than by the model. Holding features and learners fixed and varying four design choices factorially, it finds that whether training outcome windows must close before the forecast origin accounts for most of a 0.21-0.22 AUC gap between the most and least permissive designs, signed the same way in all twenty-four paired comparisons under two different outcome definitions; a placebo shows the mechanism is memorization of which countries carry a positive label; and a small audit finds the systems this literature cites as precedent do not state a closure rule. Everything else in the paper, the five-stage framework, the mobilization-versus-digital-control contest, the network decomposition, the prospective registration, is either the vehicle for that finding or does not survive it, and the manuscript now says so. What I know after reading it that I did not know before: that a window-labelled panel forecast evaluated with open training windows is a retrospective fit, that the size of the resulting inflation in this setting is about a fifth of an AUC, and that the field's own precedent systems have not guarded against it. That is a genuine, portable, and probably unwelcome result. The manuscript is honest to a fault about its limits and is still far too long for the size of its contribution.

### Part 2: Major Themes

**1. This is two papers, and only one of them is good (Edmans: contribution).** The design-sensitivity paper is a methods contribution that would change how the next autocratization forecast is evaluated. The framework paper reports a pipeline that does not beat four polyarchy variables under the design the authors themselves say is correct, and whose descriptive outputs (network dependence, betas) are shown to be seed-unstable in direction for seven of ten countries. The manuscript has been restructured to lead with the first, but the second still occupies Sections 3, 4, 5.4-5.9, most of 8, and eleven appendices, roughly 60 percent of the text. Recommendation: cut the framework to a two-page description and an appendix, and make the paper the design paper. The authors have effectively conceded this by writing "a reader who wants only a forecast should use the four variables."

**2. Two outcome definitions and no primary specification (Humphreys: measure validity).** Every strict-design result is reported twice, once on a hand-maintained 54-episode ledger that matches ERT v16 on 11 entries and once on ERT v16 itself. The paper's stated reason for not adopting ERT as primary is that the pipeline is keyed by country and cannot represent recurrent onsets without a rebuild. That is a resource constraint, not a scientific reason, and a referee will say so. The paper's own numbers make the case: ERT yields 48 distinct onsets against 28, narrower intervals throughout, and a persistence baseline whose below-chance AUC reveals a selection mechanism the ledger hides. Recommendation: state in one sentence that ERT v16 is the reference outcome for every forecasting result, report the ledger as the sensitivity, and leave the retrospective half on the ledger with the architectural caveat.

**3. The model comparison is unpowered and the paper should stop reporting it as if it could resolve anything (Nyhan 8; Blattman: absence of evidence).** The minimum detectable difference between the framework and four polyarchy variables is 0.09-0.15 of AUC, the size of the design effect itself. Table 5's six-model ladder and Table 4's watchlist comparison therefore cannot rank models, and the text repeatedly says the orderings are unresolved. Then why are they in the main text? Recommendation: keep Table 5 as the demonstration that the design collapses everyone's discrimination (which it does show), move the model-ranking prose and Table 4 to an appendix, and report the MDE once at the point where the comparison is introduced.

**4. The design audit is too small to carry the word "field" (Humphreys: external validity).** Four studies, three of them forecasting systems, coded by the authors. The V-Forecast finding is strong and stands on its own. The generalization to "the autocratization literature" rests on it plus PITF plus ViEWS, and ViEWS is shown not to be exposed. Recommendation: either code ten to fifteen published forecasts on the four dimensions, which is a week of work and would make the audit a contribution in itself, or drop "the field" and say "the three systems most often cited as precedent."

**5. Interval methods are inconsistent (Audit 5, item 1).** BCa in Table 2, percentile in Tables 3 and 5, for the same quantity, with lower bounds 0.075 apart and no reconciliation.

**6. Exposition (Evans / Bellemare).** The abstract is 459 words; no venue in this area accepts above 250. The introduction is about 1,900 words and its value-added paragraph is the third of three "findings" paragraphs, buried under a restatement of the abstract. The conclusion summarizes rather than narrates. The paper is 55 pages and roughly 25,000 words of main text against venue ceilings of 8,000-12,000. Recommendation: abstract to 200 words leading with the design result; introduction to 1,000 words with the value-added paragraph second; cut per theme 1.

### Part 3: Smaller Issues

- The five-positive training rule (Audit 5, item 2) has no robustness check and determines which origins each panel scores.
- The strict "framework" is a rank-mean blend, the conventional one a stacked meta-learner; report the single-learner strict AUCs so the reader can see the blend is not the story.
- In-sample rows in Table 2 will be quoted out of context; move them.
- The ATOP alliance channel is empty from 2024, which covers the prospective origin, so Stage 4 for the registered list runs on three edge types. This is now disclosed, and I verified that no entry in Table 7's leading-signal column names a network contribution, so nothing in the list is contaminated by it.
- The registration's tier labels contradict its own thresholds; the amendment records this. A reader of the registration file alone will still see "warning" against 0.476.
- "And not" appears 25 times in the appendices after the contrast-diversification pass; below tic density but noticeable.
- Table 1 lists GDELT coverage as 1990-2025 while Section 3.3 says meaningful coverage begins around 2000 (verified in the source). The table should carry the operative start or a note.
- Persistence at 0.344 under ERT is explained in text but the table row will still be read as a bug by a skimmer; add a table note.

### Verdict under Edmans

Contribution: strong for the design paper, weak for the framework paper. Execution: careful, unusually well falsified (placebo, factorial, era split, two outcomes, end-to-end rebuild), with the specification issues above being real but second-order. Exposition: honest, over-long, and structurally still carrying a paper it has disowned. Major revision: cut to the design paper.

---

## Round 1 follow-up: fixes applied and what they changed (2026-09-06, same day)

The author-side editor applied the Stage 4 and SDEM fixes recommended above and retrained. Findings from that pass:

**New finding, HIGH (reproducibility).** `stage5_ews/estimate.py` draws from the global numpy RNG in the critical-slowing-down surrogate generator (`np.random.normal`, lines 285 and 287) and set no numpy seed anywhere; only sklearn estimators carried `random_state`. Two identical-code runs of Stage 5 therefore produced different `eig_trend_sig`, `xcorr_trend_sig`, `mv_csd_alert` and downstream alert-tier columns (correlations 0.64-0.89 between runs on those columns). Audit 4's "seeds are correct for every stochastic component" was wrong about Stage 5: it checked estimators, not the global RNG. Fixed by seeding at module level (`np.random.seed(42 + SEED_OFFSET)`); byte-identity across two seeded runs is being verified. Every downstream number in the paper was, until this fix, reproducible only to the extent the CSD channel happened not to matter, which the paper's own stage ablation says it mostly does not.

**Stage 4 retrain (causal temporal edges, `EXCLUDE_COUNTRY` in `mask_train`, econ-similarity on raw GDP).** The 2025 contagion scores correlate 0.996 with the previous run (mean |diff| 0.0095); `network_exposure` as seen by Stage 5 correlates 0.998. The learned edge weights reordered: cultural 0.282, alliance 0.260, contiguity 0.243, econ 0.215, against the previous 0.26 / 0.25 / 0.27 / 0.22, so the manuscript's "contiguity marginally dominant" became "cultural marginally dominant" (Section 5.6 updated). Table 5 Panel A on the retrained features: framework 0.680 -> 0.678, gradient boosting 0.674 -> 0.667, elastic net 0.654 -> 0.652, random forest 0.715 -> 0.716; four polyarchy variables and persistence unchanged, as they consume no pipeline features. Paired framework-minus-four-variables -0.038 -> -0.040, interval [-0.116, +0.035]. The substantive conclusions are untouched; the third decimals are not, and the full cascade of ~23 passport-backing scripts is being rerun so every reported number matches the corrected pipeline (`run_cascade_post_stage4.sh`).

**SDEM bootstrap regenerated from its own fit (1b.3).** Lambda and rho intervals moved by bootstrap noise only. The neighbor-covariate intervals, previously centered at zero by construction, now read: y_lag [-0.003, +0.013], gdp_pc [-0.005, +0.006], urbanization [-0.007, +0.006], **trade_openness -0.0068 [-0.0131, -0.0002]**. One of the four contextual effects excludes zero. Appendix C.7's "supports nothing about contextual effects" was therefore an artifact of the mis-centered bootstrap and has been corrected in the manuscript.

**`closure_placebo.py`** now uses identical seed sets for the observed statistic and every placebo draw, N_PERM defaults to 100, and the summary reports k of n. Rerun is in the cascade.

**Status of the major concerns after this pass:** 1 (Stage 4 code) fixed and retrained, network sections being regenerated; 2 (SDEM) fixed and regenerated; 3 (hand-typed tables/figures) unchanged; 4 (replication package) partially addressed by `run_cascade_post_stage4.sh` and the tracked `quality_reports/`, README and pinning still open.

**Determinism confirmed.** Two identical-code runs of the seeded Stage 5 differ in 0 of 246 numeric columns at a 1e-9 tolerance. Remaining byte-level differences are float representation in the last digits from `n_jobs=-1` summation order and never reach a reported figure. The cascade over every `ews_signals.csv`-dependent script was launched on the confirmed-deterministic output.

**The conventional-design channel ranking does not survive the corrected pipeline.** Rerunning `channel_ablation.py` (the source of the manuscript's "removing mobilization costs six times what removing digital control costs") on three feature files isolates the cause. The ablation is deterministic given features: the block-only rows (`mob_only` 0.774, `dsp_only` 0.776) are identical across all three runs. On the committed `ews_signals.csv` the leave-one-block-out costs are mobilization -0.031 and DSP -0.008, not the -0.042 / -0.007 the manuscript reported, so the sixfold figure came from a still-earlier run of the unseeded Stage 5. Swapping only the 32 CSD-family columns into the committed file moves them to -0.025 / -0.011. On the corrected pipeline (Stage 4 fixes, Stage 5 seeded) they are **-0.005 / +0.010**: removing the mobilization block costs almost nothing and removing the digital-control block improves out-of-sample AUC. The directional decomposition (Table 20) collapses with it (pro-autocratic -0.021 -> +0.004, pro-democratic -0.009 -> -0.000), the Rashomon-set count falls from 21/30 to 13/30, and the measurement-uncertainty sweep now reads 30/30 in the other direction of what it meant before (the ordering it preserved is one the corrected pipeline does not produce). The conventional design therefore does not separate the channels any more than the strict design does. The manuscript's abstract, introduction, Section 5.4, Section 8.1, conclusion, scope paragraph, Appendix A.1 and Table 20 all carried the sixfold claim and are being rewritten; the design finding (Table 6, placebo, size-matched arm) is untouched by this and, if anything, is the explanation for it.

Other retrospective figures moved with the seeding: 2019 hold-out AUC-PR 0.591 -> 0.622 with the Brier skill interval now excluding zero ([0.013, ...]), so "no demonstrated probabilistic skill" reverses; lead-time AUCs 0.956/0.903/0.854/0.835 -> 0.949/0.910/0.864/0.858; LOEO warning and alert tiers 22 -> 18 and 12 -> 11; reliability bins reshuffled (top bin 6 -> 10 country-years, observed 0.833 -> 0.600). The strict-design figures moved at the third decimal only.
