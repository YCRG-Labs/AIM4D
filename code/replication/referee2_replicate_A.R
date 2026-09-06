#!/usr/bin/env Rscript
#
# Referee 2 / Audit 2 (Cross-Language Replication) -- TARGET A
#
# Independently re-derives Table 5 Panel A (strict comparison, our episode
# set) from the saved predictions CSV, without touching any author file.
# AUC-ROC and average precision are implemented by hand from their
# definitions and matched against sklearn's exact algorithms (rank-based
# Mann-Whitney U for AUC; grouped-threshold precision/recall differencing
# for AP), not against any R package, per the audit brief.
#
# Input:  robustness/strict_table_predictions_h5.csv
# Output: printed comparison table only (no files written under AIM4D/).

args <- commandArgs(trailingOnly = FALSE)
here <- normalizePath(dirname(sub("^--file=", "", args[grep("^--file=", args)])))
root <- normalizePath(file.path(here, "..", ".."))
infile <- file.path(root, "robustness", "strict_table_predictions_h5.csv")

df <- read.csv(infile, stringsAsFactors = FALSE)
stopifnot(all(c("country_name", "year", "y", "p", "model") %in% names(df)))

## ---- sklearn-exact AUC-ROC via Mann-Whitney U with mid-rank ties --------
sklearn_auc <- function(y, p) {
  n_pos <- sum(y == 1)
  n_neg <- sum(y == 0)
  r <- rank(p, ties.method = "average")
  u <- sum(r[y == 1]) - n_pos * (n_pos + 1) / 2
  u / (n_pos * n_neg)
}

## ---- sklearn-exact average precision (grouped-threshold, matches ties) --
sklearn_ap <- function(y, p) {
  n <- length(y)
  ord <- order(-p)
  ys <- y[ord]
  ps <- p[ord]
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

models <- unique(df$model)
cat("Models found (", length(models), "):\n", sep = "")
for (m in models) cat("  -", m, "\n")
cat("\n")

## ---- structural checks ---------------------------------------------------
key_sets <- lapply(models, function(m) {
  sub <- df[df$model == m, c("country_name", "year")]
  paste(sub$country_name, sub$year, sep = "||")
})
n_per_model <- sapply(key_sets, length)
identical_sets <- all(sapply(key_sets, function(s) setequal(s, key_sets[[1]])))
n_pos_per_model <- sapply(models, function(m) sum(df$y[df$model == m]))
year_range <- range(df$year)

cat("Structural checks:\n")
cat(sprintf("  n rows per model: %s (expect all 933)\n",
            paste(unique(n_per_model), collapse = ",")))
cat(sprintf("  n_pos per model : %s (expect all 107)\n",
            paste(unique(n_pos_per_model), collapse = ",")))
cat(sprintf("  identical (country,year) sets across all 6 models: %s\n", identical_sets))
cat(sprintf("  year range: %d..%d (expect 2008..2020)\n\n", year_range[1], year_range[2]))

## ---- per-model AUC / AP ---------------------------------------------------
results <- data.frame(model = character(), n = integer(), n_pos = integer(),
                       auc = numeric(), ap = numeric(), stringsAsFactors = FALSE)
for (m in models) {
  sub <- df[df$model == m, ]
  auc <- sklearn_auc(sub$y, sub$p)
  ap <- sklearn_ap(sub$y, sub$p)
  results <- rbind(results, data.frame(model = m, n = nrow(sub), n_pos = sum(sub$y),
                                        auc = round(auc, 3), ap = round(ap, 3)))
}

cat("Per-model AUC / AP (R replication):\n")
print(results, row.names = FALSE)

## ---- machine-readable summary for the report -----------------------------
cat("\n---AUDIT2_TARGET_A_JSON---\n")
for (i in seq_len(nrow(results))) {
  cat(sprintf('{"model":"%s","n":%d,"n_pos":%d,"auc":%.4f,"ap":%.4f}\n',
              results$model[i], results$n[i], results$n_pos[i],
              results$auc[i], results$ap[i]))
}
cat(sprintf('{"check":"n_per_model","values":"%s"}\n', paste(unique(n_per_model), collapse=",")))
cat(sprintf('{"check":"n_pos_per_model","values":"%s"}\n', paste(unique(n_pos_per_model), collapse=",")))
cat(sprintf('{"check":"identical_key_sets","value":%s}\n', tolower(as.character(identical_sets))))
cat(sprintf('{"check":"year_range","min":%d,"max":%d}\n', year_range[1], year_range[2]))
