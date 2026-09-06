#!/usr/bin/env Rscript
#
# Referee 2 / Audit 2 (Cross-Language Replication) -- TARGET C
#
# Re-derives the Table 6 factorial marginals (each design dimension's
# average AUC effect, holding the other three dimensions and the learner
# fixed) from the saved design_factorial CSVs, independently in R.
#
# Input:  robustness/design_factorial.csv
#         robustness/design_factorial_ert.csv
# Output: printed comparison only (no files written under AIM4D/).

args <- commandArgs(trailingOnly = FALSE)
here <- normalizePath(dirname(sub("^--file=", "", args[grep("^--file=", args)])))
root <- normalizePath(file.path(here, "..", ".."))

## stricter level for each dimension (the level the author's reported arrow
## points TO, e.g. "all -> at-risk" means "at-risk" is the stricter level)
dims <- list(
  risk_set = list(laxer = "all",        stricter = "at-risk",     label = "risk_set (all->at-risk)"),
  label    = list(laxer = "window",     stricter = "future-only", label = "label (window->future-only)"),
  origin   = list(laxer = "fixed-2019", stricter = "rolling",     label = "origin (fixed-2019->rolling)"),
  closure  = list(laxer = "none",       stricter = "enforced",    label = "closure (none->enforced)")
)

dim_names <- names(dims)

analyze <- function(path, tag) {
  df <- read.csv(path, stringsAsFactors = FALSE)
  stopifnot(nrow(df) == 48)
  cat(sprintf("\n=== %s (%s) ===\n", tag, path))
  cat(sprintf("rows: %d (expect 48 = 16 cells x 3 learners)\n\n", nrow(df)))

  results <- list()
  for (dname in dim_names) {
    other_dims <- setdiff(dim_names, dname)
    lax_lvl <- dims[[dname]]$laxer
    str_lvl <- dims[[dname]]$stricter

    diffs <- c()
    for (lrn in unique(df$learner)) {
      sub_all <- df[df$learner == lrn, ]
      # enumerate the 4 combinations of the other three dims present in data
      combos <- unique(sub_all[, other_dims])
      for (i in seq_len(nrow(combos))) {
        combo <- combos[i, , drop = FALSE]
        match_mask <- Reduce(`&`, lapply(other_dims, function(od) sub_all[[od]] == combo[[od]]))
        cell_lax <- sub_all[match_mask & sub_all[[dname]] == lax_lvl, ]
        cell_str <- sub_all[match_mask & sub_all[[dname]] == str_lvl, ]
        if (nrow(cell_lax) == 1 && nrow(cell_str) == 1) {
          diffs <- c(diffs, cell_str$auc - cell_lax$auc)
        }
      }
    }
    results[[dname]] <- diffs
    cat(sprintf("%-35s n_pairs=%d  mean=%+.3f  [min=%+.3f, max=%+.3f]\n",
                dims[[dname]]$label, length(diffs), mean(diffs), min(diffs), max(diffs)))
  }

  ## fully conventional / fully strict corner cells, averaged over learners
  conv <- df[df$risk_set == "all" & df$label == "window" &
             df$origin == "fixed-2019" & df$closure == "none", ]
  strict <- df[df$risk_set == "at-risk" & df$label == "future-only" &
               df$origin == "rolling" & df$closure == "enforced", ]
  cat(sprintf("\nfully conventional cell (all/window/fixed-2019/none), n=%d learners: mean AUC = %.3f\n",
              nrow(conv), mean(conv$auc)))
  cat(sprintf("fully strict cell (at-risk/future-only/rolling/enforced), n=%d learners: mean AUC = %.3f\n",
              nrow(strict), mean(strict$auc)))

  list(results = results, conv = mean(conv$auc), strict = mean(strict$auc))
}

r1 <- analyze(file.path(root, "robustness", "design_factorial.csv"), "our episode set")
r2 <- analyze(file.path(root, "robustness", "design_factorial_ert.csv"), "ERT episode set")

cat("\n---AUDIT2_TARGET_C_JSON---\n")
emit <- function(tag, r) {
  for (dname in names(r$results)) {
    d <- r$results[[dname]]
    cat(sprintf('{"set":"%s","dim":"%s","mean":%.4f,"min":%.4f,"max":%.4f,"n_pairs":%d}\n',
                tag, dname, mean(d), min(d), max(d), length(d)))
  }
  cat(sprintf('{"set":"%s","corner":"conventional","auc":%.4f}\n', tag, r$conv))
  cat(sprintf('{"set":"%s","corner":"strict","auc":%.4f}\n', tag, r$strict))
}
emit("own", r1)
emit("ert", r2)
