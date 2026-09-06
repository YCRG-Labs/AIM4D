# Amendment to the 2026-05-17 Prospective Registration

**Amendment date:** 2026-09-06
**Amends:** `PROSPECTIVE_FORECAST_2026-2031.md` (registered 2026-05-17, commit `a360f8a`)
**Status of the original:** unchanged. That file has exactly one commit in the
repository history and has never been edited. Nothing here rewrites it.

This amendment exists because the manuscript and the original registration had
drifted apart. Rather than quietly align the registration to the paper, which
would destroy the thing a registration is for, the divergences are recorded
here with dates. The registered list of 25 countries and their risk scores are
**unchanged** and remain the object to be scored in 2031.

## What is unchanged

- The 25 countries, their risk scores, and their rank order.
- The five falsification criteria (1)-(5) defining a true positive.
- The training cutoff, the 46 training episodes, and the model commit.

## Divergences between the registration and the manuscript

### 1. The scores are not calibrated probabilities

The registration states that "Risk is a calibrated probability of
autocratization onset between 2026 and 2031." Subsequent calibration analysis
reported in the manuscript shows this is wrong. On the held-out slice the
calibration slope is 3.40 with a country-clustered interval of [2.45, 5.26] and
the intercept is 2.26 [1.05, 4.04], both excluding the ideal values of one and
zero. The scores order countries well and do not carry probability meaning: a
score of 0.30 does not mean a thirty percent chance.

**Consequence for scoring.** The Brier score listed in the registration's
metric table is not a meaningful check on the registered values and should be
computed, if at all, only after the prequential recalibration described in the
manuscript. Rank-based criteria are unaffected, because the tiers are
percentile ranks and percentile ranks are invariant to monotone rescaling.

### 2. The registration's tier labels contradict its own thresholds

The registration fixes the tiers at watch = 0.075, warning = 0.20, alert = 0.40,
then labels Bolivia (0.476), Argentina (0.463) and Niger (0.456) as "warning"
and Iran (0.409) and Uganda (0.409) as "watch". All five sit above the stated
alert threshold of 0.40. The tier *labels* in the registration table are
therefore internally inconsistent with the tier *thresholds* in the same file.

The manuscript reports these five countries as alert-tier, which is what the
registered thresholds imply. We treat the numeric thresholds as the registered
quantity and the labels as a transcription error in the original file, and we
record it here rather than correcting the frozen document.

### 3. The registration locks no evaluation thresholds

The registration's metric table lists Precision@5, @10, @25, Brier and a
warning-tier hit rate, with the "Locked threshold" column empty for every row.
No pass/fail bar was actually fixed on 2026-05-17.

The manuscript states two criteria that are **not** in the registration and are
therefore **registered as of this amendment, 2026-09-06, not 2026-05-17**:

- **(A)** Precision among the nine novel candidates, meaning the countries with
  no episode anywhere in the training set, must exceed the historical novel
  base rate of approximately 0.17 at matched list depth.
- **(B)** The Spearman correlation between the full 2026 risk ranking over all
  evaluated countries and the realized magnitude of democratic decline across
  2026-2031.

Both carry the later date and must be reported as such. The date deserves
precision rather than the phrase "before any outcome is observed", which would
be false: 2026-09-06 falls *inside* the 2026-2031 window, not before it. What
is true is narrower. None of the sources the registration names had published
coding for 2026 at that date, and no outcome data were consulted in setting
these criteria. That is absence of finalized annual coding, not absence of
knowledge of events in the first eight months of 2026, and we do not claim the
latter. Five of the window's six years lay entirely in the future; the sixth did
not. Criterion (A) is therefore weaker on timing grounds than the registered
list itself, which was locked in May 2026 before the window opened. The
manuscript's power analysis for criterion (A) applies: on nine candidates the
bar is cleared by two onsets, which under a true precision of 0.17 occurs by
chance with probability 0.47, so clearing it is weak evidence of skill.

### 4. Horizon and eligible population

The registration states a six-year horizon (2026-2031). The model scores a
five-year window. The registered horizon is one year longer than the estimand,
and the eligible population behind the registered list differs from both the
strict-design risk set (democracies outside an episode) and the retrospective
novel-candidate set. These three populations carry different base rates and are
not interchangeable. The registered list should be read as a commitment device
for this specific list, not as a second estimate of the quantity the strict
design reports.

### 5. Headline validation figure

The registration reports a headline out-of-sample AUC of 0.933 with a
cluster-bootstrap interval of [0.850, 0.983] for a 2020-2025 validation. The
manuscript reports 0.939 [0.800, 0.975] on the 2019 temporal hold-out under the
current pipeline. The difference reflects the beta-winsorization correction
applied after registration and the bootstrap being recomputed; neither figure is
the paper's primary result, which is the strict forecasting estimate.

### 6. Author list

The registration names "Crainic, Yee & Sharma". The manuscript is authored by
Crainic and Yee. The registration's author list is superseded.

### 7. External timestamp

The registration's "External timestamp" section contains placeholder URLs for
OSF and arXiv rather than deposited links. **No independent external timestamp
of the registration exists.** The only timestamp is the repository's own git
history, commit `a360f8a`, dated 2026-05-18 00:55 EDT, which is a self-hosted
record and not third-party verification. This should be stated plainly wherever
the registration is described, and we do not claim an independent timestamp the
registration does not have.

## Scoring instruction

When this forecast is scored after 2031, report:

1. The registered top-25 list and its outcomes under criteria (1)-(5), which
   carry the 2026-05-17 date.
2. Criteria (A) and (B) above, flagged as carrying the 2026-09-06 date.
3. The corrected-pipeline ranking alongside the registered one, clearly
   labelled as secondary, as the manuscript commits to doing.

All three are to be published however they fall.
