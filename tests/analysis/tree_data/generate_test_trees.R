library(ape)
library(treestats)

set.seed(42)

n_trees <- 10
n_tips <- 50

results <- data.frame(
  tree_id = integer(),
  colless = numeric(),
  phylogenetic_diversity = numeric(),
  cherries = numeric(),
  laplace_spectrum_principal_eigenvalue = numeric(),
  laplace_spectrum_eigengap = numeric(),
  laplace_spectrum_asymmetry = numeric(),
  laplace_spectrum_peakedness = numeric()
)

for (i in 1:n_trees) {
  tree <- rphylo(n = n_tips, birth = 1, death = 0.5)

  newick_file <- sprintf("tree_%02d.newick", i)
  write.tree(tree, file = newick_file)

  colless_stat <- colless(tree)
  pd_stat <- phylogenetic_diversity(tree)
  cherries_stat <- cherries(tree)

  laplace_spec <- laplacian_spectrum(tree)
  laplace_principal <- laplace_spec$principal_eigenvalue
  laplace_eigengap <- laplace_spec$eigengap[[1]]
  laplace_asymmetry <- laplace_spec$asymmetry
  laplace_peakedness <- laplace_spec$peakedness

  results[i, ] <- list(i, colless_stat, pd_stat, cherries_stat,
                       laplace_principal, laplace_eigengap, laplace_asymmetry, laplace_peakedness)
}

csv_file <- "tree_statistics.csv"
write.csv(results, csv_file, row.names = FALSE)

cat("Generated", n_trees, "trees with seed 42\n")
cat("Trees and statistics saved to current directory\n\n")
print(results)
