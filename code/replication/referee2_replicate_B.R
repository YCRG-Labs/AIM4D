#!/usr/bin/env Rscript
#
# Referee 2 / Audit 2 (Cross-Language Replication) -- TARGET B
#
# Re-derives the paired country-clustered bootstrap difference (framework
# minus four polyarchy variables) from the saved predictions CSV, using an
# independently written R implementation of the author's paired_ci()
# procedure (robustness/strict_table_final.py) with a FRESH R random seed.
#
# The author's Python RNG (np.random.default_rng(20260905)) is advanced by
# six earlier per-model boot_ci() calls (2000 draws each) before paired_ci()
# consumes its own 2000 draws, so a fresh-seeded run in any language cannot
# reproduce the author's exact reported interval bit-for-bit. This script
# instead checks: (i) does a fresh-seeded R bootstrap land within Monte
# Carlo error of the author's interval, and (ii) does the point estimate
# (which is not seed-sensitive at this N_BOOT) match -0.037/-0.038.
#
# Input:  robustness/strict_table_predictions_h5.csv
# Output: printed comparison only (no files written under AIM4D/).

args <- commandArgs(trailingOnly = FALSE)
here <- normalizePath(dirname(sub("^--file=", "", args[grep("^--file=", args)])))
root <- normalizePath(file.path(here, "..", ".."))
infile <- file.path(root, "robustness", "strict_table_predictions_h5.csv")

df <- read.csv(infile, stringsAsFactors = FALSE)

sklearn_auc <- function(y, p) {
  n_pos <- sum(y == 1); n_neg <- sum(y == 0)
  r <- rank(p, ties.method = "average")
  (sum(r[y == 1]) - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)
}

sklearn_ap <- function(y, p) {
  n <- length(y)
  ord <- order(-p)
  ys <- y[ord]; ps <- p[ord]
  d <- which(diff(ps) != 0)
  thresh_idx <- c(d, n)
  cum_tp <- cumsum(ys)
  tps <- cum_tp[thresh_idx]
  total_pos <- sum(ys)
  precision <- tps / thresh_idx
  recall <- tps / total_pos
  recall_prev <- c(0, recall[-length(recall)])
  sum((recall - recall_prev) * precision)
}

fw_name <- "Five-stage framework, rank-mean blend"
pv_name <- "Four polyarchy variables"

fw <- df[df$model == fw_name, ]
pv <- df[df$model == pv_name, ]

fw <- fw[order(fw$country_name, fw$year), ]
pv <- pv[order(pv$country_name, pv$year), ]

stopifnot(nrow(fw) == nrow(pv))
stopifnot(all(fw$country_name == pv$country_name))
stopifnot(all(fw$year == pv$year))
stopifnot(all(fw$y == pv$y))

y <- fw$y
pa <- fw$p          # framework
pb <- pv$p          # four polyarchy variables
countries <- fw$country_name

N_BOOT <- 2000
uniq <- sort(unique(countries))
idx <- setNames(lapply(uniq, function(c) which(countries == c)), uniq)

paired_ci_R <- function(seed) {
  set.seed(seed)
  da <- numeric(0); dp <- numeric(0)
  for (b in seq_len(N_BOOT)) {
    draw <- sample(uniq, size = length(uniq), replace = TRUE)
    j <- unlist(idx[draw], use.names = FALSE)
    ysum <- sum(y[j])
    if (ysum < 3 || ysum == length(j)) next
    da <- c(da, sklearn_auc(y[j], pa[j]) - sklearn_auc(y[j], pb[j]))
    dp <- c(dp, sklearn_ap(y[j], pa[j]) - sklearn_ap(y[j], pb[j]))
  }
  list(
    n_valid = length(da),
    auc = c(mean = round(mean(da), 4),
            lo = round(as.numeric(quantile(da, 0.025, type = 7)), 4),
            hi = round(as.numeric(quantile(da, 0.975, type = 7)), 4)),
    ap = c(mean = round(mean(dp), 4),
           lo = round(as.numeric(quantile(dp, 0.025, type = 7)), 4),
           hi = round(as.numeric(quantile(dp, 0.975, type = 7)), 4))
  )
}

cat(sprintf("N rows: %d   N unique countries: %d\n\n", nrow(fw), length(uniq)))

seeds <- c(1, 42, 20260905, 777, 2024)
cat("Fresh-seed paired bootstrap replications (R, independent of author's Python RNG stream):\n\n")
for (s in seeds) {
  res <- paired_ci_R(s)
  cat(sprintf("seed=%-10d n_valid=%d  dAUC mean=%+.4f [%+.4f, %+.4f]   dAP mean=%+.4f [%+.4f, %+.4f]\n",
              s, res$n_valid,
              res$auc["mean"], res$auc["lo"], res$auc["hi"],
              res$ap["mean"], res$ap["lo"], res$ap["hi"]))
}

cat("\nAuthor (Python, seed 20260905, RNG advanced by 6 prior model-level boot_ci calls):\n")
cat("  dAUC -0.038 [-0.116, +0.039]   dAP -0.026 [-0.125, +0.070]\n")
cat("Author's own fresh-seeded default_rng(20260905) check (stated in audit brief):\n")
cat("  dAUC [-0.113, +0.040]\n")

cat("\n---AUDIT2_TARGET_B_JSON---\n")
for (s in seeds) {
  res <- paired_ci_R(s)
  cat(sprintf('{"seed":%d,"n_valid":%d,"dauc_mean":%.4f,"dauc_lo":%.4f,"dauc_hi":%.4f,"dap_mean":%.4f,"dap_lo":%.4f,"dap_hi":%.4f}\n',
              s, res$n_valid, res$auc["mean"], res$auc["lo"], res$auc["hi"],
              res$ap["mean"], res$ap["lo"], res$ap["hi"]))
}
