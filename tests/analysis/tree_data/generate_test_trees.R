library(ape)
library(treestats)

set.seed(42)

n_trees <- 10
n_tips <- 50

results <- data.frame(
  tree_id = integer(),
  colless = numeric(),
  phylogenetic_diversity = numeric(),
  cherries = numeric()
)

for (i in 1:n_trees) {
  tree <- rphylo(n = n_tips, birth = 1, death = 0.5)

  newick_file <- sprintf("tree_%02d.newick", i)
  write.tree(tree, file = newick_file)

  colless_stat <- colless(tree)
  pd_stat <- phylogenetic_diversity(tree)
  cherries_stat <- cherries(tree)

  results[i, ] <- list(i, colless_stat, pd_stat, cherries_stat)
}

csv_file <- "tree_statistics.csv"
write.csv(results, csv_file, row.names = FALSE)

cat("Generated", n_trees, "trees with seed 42\n")
cat("Trees and statistics saved to current directory\n\n")
print(results)
