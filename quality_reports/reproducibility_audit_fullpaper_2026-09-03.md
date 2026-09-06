# Reproducibility Audit: Forecasting the Onset of Autocratization

**Date:** 2026-09-03
**Manuscript:** ~/Downloads/fullpaper (2).tex
**Passport:** quality_reports/passports/aim4d-fullpaper.yaml (23 claims)
**Method:** /audit-reproducibility in passport mode, via robustness/audit_claims.py
**Tolerance source:** per-claim tolerances declared in the passport (point estimates 0.005 unless stated; counts exact), tighter than the protocol defaults

## Summary

| Status | Count |
|---|---|
| PASS | 101 |
| FAIL | 0 |
| EXPLAINED | 0 |
| UNMATCHED | 0 |
| Reported value absent from the .tex | 0 |
| **Overall verdict** | **PASS** |

`PASS 101   FAIL 0   UNMATCHED 0   value-absent-from-tex 0   total 101`

Run after the 2026-09-02/03 referee pass, which rewrote the abstract,
introduction, Sections 3.1, 4.3, 5.1, 6.1, 8.1 and 8.3, the conclusion, Figure
3, Table 13, and added Table 16. Each check recomputes the value from the named
output file and separately confirms the figure appears in the manuscript, which
catches an output and a passport agreeing while the paper goes stale.

## Resolved this pass

**C22, the weighted-kappa discrepancy, is fixed at source rather than disclosed.**
The old `hmm_states.py` sweep differed from the pipeline in four ways: a full
rather than diagonal covariance, twenty rather than sixty restarts, no
stabilization filter, and a fit on the whole panel instead of the pre-2019
training window. `hmm_states_locked.py` re-runs it under the locked
specification and returns 0.718 at S=5, reproducing the headline 0.72. It also
strengthens the state-count argument: under the locked specification both the
weighted and the unweighted kappa peak at S=5 and fall at S=6, where the old
sweep had weighted kappa climbing monotonically to S=6 and so appeared to argue
for more states than the paper uses. Table 13 now reports the locked sweep.

## Two auditor bugs found and fixed (not paper defects)

Recorded because they are the failure mode the protocol warns about, where the
computed value is a challenger rather than an oracle.

1. `network_seed_sweep_summary.csv` is indexed by statistic (mean/std/min/max),
   not by seed. Averaging every row folded standard deviations and extrema into
   the mean and produced four spurious C13 FAILs. The `mean` row reproduces the
   paper exactly.
2. `contagion_seed_sweep_summary.csv` stores the country as `Türkiye` in NFD
   form, so an ASCII substring search for "Turkey" missed it and reported C14 as
   UNMATCHED. Normalising before matching resolves it.

## All checks

```
claim                             what  reported  computed   diff    tol  num in_tex
   C1                refit CV mean AUC     0.821     0.821 0.0000  0.005 PASS    yes
   C1                 refit CV mean AP     0.524     0.524 0.0000  0.005 PASS    yes
   C2           auc_roc_oos_2019 point     0.934     0.934 0.0000  0.005 PASS    yes
   C2           auc_roc_oos_2019 CI lo     0.776     0.776 0.0000   0.01 PASS    yes
   C2           auc_roc_oos_2019 CI hi     0.977     0.977 0.0000   0.01 PASS    yes
   C2            auc_pr_oos_2019 point     0.656     0.656 0.0000  0.005 PASS    yes
   C2            auc_pr_oos_2019 CI lo     0.434     0.434 0.0000   0.01 PASS    yes
   C2            auc_pr_oos_2019 CI hi     0.816     0.816 0.0000   0.01 PASS    yes
   C2               bss_oos_2019 point     0.231     0.231 0.0000  0.005 PASS    yes
   C2               bss_oos_2019 CI lo     0.041     0.041 0.0000   0.01 PASS    yes
   C2               bss_oos_2019 CI hi     0.344     0.344 0.0000   0.01 PASS    yes
   C7     auc_fh_3yr_decline_2pt point     0.770     0.770 0.0000  0.005 PASS    yes
   C7     auc_fh_3yr_decline_2pt CI lo     0.722     0.722 0.0000   0.01 PASS    yes
   C7     auc_fh_3yr_decline_2pt CI hi     0.833     0.833 0.0000   0.01 PASS    yes
   C7 auc_polity_3yr_decline_3pt point     0.743     0.743 0.0000  0.005 PASS    yes
   C7 auc_polity_3yr_decline_3pt CI lo     0.694     0.694 0.0000   0.01 PASS    yes
   C7 auc_polity_3yr_decline_3pt CI hi     0.780     0.780 0.0000   0.01 PASS    yes
   C3                       lead 1 AUC     0.956     0.956 0.0000  0.005 PASS    yes
   C3                       lead 2 AUC     0.903     0.903 0.0000  0.005 PASS    yes
   C3                       lead 3 AUC     0.854     0.854 0.0000  0.005 PASS    yes
   C3                       lead 4 AUC     0.835     0.835 0.0000  0.005 PASS    yes
   C4                       LOEO watch    32.000    32.000  exact  exact PASS    yes
   C4                     LOEO warning    18.000    18.000  exact  exact PASS    yes
   C4                       LOEO alert    12.000    12.000  exact  exact PASS    yes
   C4             backsliding detected    22.000    22.000  exact  exact PASS    yes
   C4                    coup detected    10.000    10.000  exact  exact PASS    yes
   C5            15-ep LOEO mean delta     0.081     0.081 0.0000  0.005 PASS    yes
   C5               15-ep higher count    14.000    14.000  exact  exact PASS    yes
   C6                  booster AUC raw     0.940     0.940 0.0000  0.005 PASS    yes
   C6               booster AUC-PR raw     0.735     0.735 0.0000  0.005 PASS    yes
   C6             booster AUC-PR clean     0.634     0.634 0.0000  0.005 PASS    yes
   C8         perm imp f1_rolling_mean     0.019     0.019 0.0000  0.001 PASS    yes
   C8               perm imp f1_change     0.013     0.013 0.0000  0.001 PASS    yes
   C8              perm imp v2cademmob     0.010     0.010 0.0000  0.001 PASS    yes
   C8    perm imp v2cagenmob_detrended     0.009     0.009 0.0000  0.001 PASS    yes
   C8              perm imp v2smgovdom     0.008     0.008 0.0000  0.001 PASS    yes
   C9            mobilization mean imp     0.069     0.069 0.0000  0.005 PASS    yes
   C9         digital control mean imp     0.063     0.063 0.0000  0.005 PASS    yes
   C9                   mob > dsp rows    21.000    21.000  exact  exact PASS    yes
   C9                mob > factor rows     0.000     0.000  exact  exact PASS    yes
  C10                     mob z at t-5     0.190     0.190 0.0000   0.01 PASS    yes
  C10                     mob z at t-1     0.470     0.470 0.0000   0.01 PASS    yes
  C10                        mob_fires    25.000    25.000  exact  exact PASS    yes
  C10                        dig_fires     8.000     8.000  exact  exact PASS    yes
  C11                     DSP full OOS     0.905     0.905 0.0000  0.005 PASS    yes
  C11                  DSP ablated OOS     0.899     0.899 0.0000  0.005 PASS    yes
  C11                     DSP only OOS     0.774     0.774 0.0000  0.005 PASS    yes
  C13               sweep alpha_contig     0.240     0.240 0.0000   0.01 PASS    yes
  C13             sweep alpha_alliance     0.260     0.260 0.0000   0.01 PASS    yes
  C13                sweep alpha_trade     0.220     0.220 0.0000   0.01 PASS    yes
  C13             sweep alpha_cultural     0.280     0.280 0.0000   0.01 PASS    yes
  C14           Hungary contagion mean     0.606     0.606 0.0000  0.005 PASS    yes
  C14             Hungary contagion sd     0.064     0.064 0.0000   0.01 PASS    yes
  C14            Turkey contagion mean     0.323     0.323 0.0000  0.005 PASS    yes
  C14              Turkey contagion sd     0.037     0.037 0.0000   0.01 PASS    yes
  C15                     vdem unc AUC     0.920     0.920 0.0000   0.01 PASS    yes
  C15                      vdem unc sd     0.010     0.010 0.0000  0.005 PASS    yes
  C15                    mob>dsp draws    29.000    29.000  exact  exact PASS    yes
  C16                    enet selected    89.000    89.000  exact  exact PASS    yes
  C16                     enet OOS AUC     0.932     0.932 0.0000  0.005 PASS    yes
  C16                  enet OOS AUC-PR     0.602     0.602 0.0000  0.005 PASS    yes
  C17                  perm congruence     0.060     0.060 0.0000   0.01 PASS    yes
  C17                   hmm kappa real     0.580     0.580 0.0000   0.01 PASS    yes
  C17              hmm kappa scrambled    -0.010    -0.010 0.0000   0.01 PASS    yes
  C19                     mob-only AUC     0.716     0.716 0.0000  0.005 PASS    yes
  C19                  mob-only AUC-PR     0.151     0.151 0.0000  0.005 PASS    yes
  C20           reliability bin 1 pred     0.091     0.091 0.0000  0.005 PASS    yes
  C20            reliability bin 1 obs     0.009     0.009 0.0000  0.005 PASS    yes
  C20              reliability bin 1 n   464.000   464.000  exact  exact PASS    yes
  C20           reliability bin 2 pred     0.183     0.183 0.0000  0.005 PASS    yes
  C20            reliability bin 2 obs     0.044     0.044 0.0000  0.005 PASS    yes
  C20              reliability bin 2 n   203.000   203.000  exact  exact PASS    yes
  C20           reliability bin 3 pred     0.309     0.309 0.0000  0.005 PASS    yes
  C20            reliability bin 3 obs     0.500     0.500 0.0000  0.005 PASS    yes
  C20              reliability bin 3 n    40.000    40.000  exact  exact PASS    yes
  C20           reliability bin 4 pred     0.423     0.423 0.0000  0.005 PASS    yes
  C20            reliability bin 4 obs     0.696     0.696 0.0000  0.005 PASS    yes
  C20              reliability bin 4 n    23.000    23.000  exact  exact PASS    yes
  C20           reliability bin 5 pred     0.549     0.549 0.0000  0.005 PASS    yes
  C20            reliability bin 5 obs     0.833     0.833 0.0000  0.005 PASS    yes
  C20              reliability bin 5 n     6.000     6.000  exact  exact PASS    yes
  C21                        dsp d_AUC    -0.007    -0.006 0.0010  0.005 PASS    yes
  C21                     dsp d_AUC_PR    -0.010    -0.010 0.0000  0.005 PASS    yes
  C21                     dsp_only OOS     0.776     0.776 0.0000  0.005 PASS    yes
  C21                        mob d_AUC    -0.042    -0.042 0.0000  0.005 PASS    yes
  C21                     mob d_AUC_PR    -0.097    -0.097 0.0000  0.005 PASS    yes
  C21                     mob_only OOS     0.774     0.774 0.0000  0.005 PASS    yes
  C21                     factor d_AUC    -0.033    -0.033 0.0000  0.005 PASS    yes
  C21                  factor d_AUC_PR    -0.146    -0.146 0.0000  0.005 PASS    yes
  C21                  factor_only OOS     0.843     0.843 0.0000  0.005 PASS    yes
  C22               locked kappa (K=4)     0.720     0.720 0.0000   0.01 PASS    yes
  C22         locked sweep S=3 kappa_w     0.574     0.574 0.0000  0.005 PASS    yes
  C22           locked sweep S=3 kappa     0.444     0.444 0.0000  0.005 PASS    yes
  C22         locked sweep S=4 kappa_w     0.647     0.647 0.0000  0.005 PASS    yes
  C22           locked sweep S=4 kappa     0.479     0.479 0.0000  0.005 PASS    yes
  C22         locked sweep S=5 kappa_w     0.718     0.718 0.0000  0.005 PASS    yes
  C22           locked sweep S=5 kappa     0.500     0.500 0.0000  0.005 PASS    yes
  C22         locked sweep S=6 kappa_w     0.699     0.699 0.0000  0.005 PASS    yes
  C22           locked sweep S=6 kappa     0.387     0.387 0.0000  0.005 PASS    yes
  C22   locked S=5 reproduces pipeline     0.720     0.720 0.0000   0.01 PASS    yes
  C22                     sanity kappa     0.580     0.580 0.0000   0.01 PASS    yes
```

## Environment

Python 3.14 with numpy, pandas and scikit-learn 1.9.0 in `.venv_ablation`. That
scikit-learn differs from the July 2026 certified run, which is the source of
the sub-0.002 drift between `dsp_ablation.csv` (C11) and `channel_ablation.csv`
(C21) on their three shared configurations. Both sit inside C11's declared 0.005
tolerance and both are cited in the paper.

## Next steps

1. No FAILs to resolve.
2. C18's registered forecast CSV remains frozen and must never be regenerated.
   Note that the registered list predates the Stage-2 beta winsorization fix;
   the paper discloses this and reports 23 of 25 countries retained.
3. `hmm_states.py` and `hmm_states_results.csv` are kept as superseded
   provenance and are no longer cited in the manuscript.
