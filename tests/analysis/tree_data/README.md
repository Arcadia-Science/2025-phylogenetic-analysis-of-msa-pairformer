# Tree statistics test data

This directory contains test data for validating Python implementations of phylogenetic tree statistics. The data serves as ground truth for testing Python implementations of tree statistics. The ground truth values were calculated using the R package **treestats**, which is the reference implementation for these metrics.

## Data Generation

The test data was generated using the R script `generate_test_trees.R`, which:

1. Generates 10 random phylogenetic trees using `ape::rphylo` with:
   - 50 tips per tree
   - Birth rate = 1
   - Death rate = 0.5
   - Random seed = 42 (for reproducibility)

2. Saves each tree in Newick format to `tree_{N}.newick`

3. Calculates three statistics for each tree using the `treestats` package:
   - **Colless index**: Measures tree balance
   - **Phylogenetic diversity**: Sum of all branch lengths
   - **Cherries**: Number of cherry structures (pairs of sister tips)

4. Saves all statistics to `tree_statistics.csv`. These are considered the ground truth values that the Python implementations are compared against.

## Reproducibility

To regenerate the test data, run from this directory:

```bash
cd tests/analysis/tree_data
Rscript generate_test_trees.R
```

The script uses `set.seed(42)` to ensure reproducible results across runs.

## Ground Truth

The R package **treestats** is used as the ground truth for all metric calculations. Python implementations should match these values to validate correctness.

## Directory Structure

```
tests/analysis/tree_data/
├── README.md
├── generate_test_trees.R
├── tree_1.newick
├── tree_2.newick
├── ...
├── tree_10.newick
└── tree_statistics.csv
```
