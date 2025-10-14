from pathlib import Path
from typing import Protocol

import pandas as pd
import pytest
from analysis.tree import (
    cherry_count_statistic,
    colless_statistic,
    phylogenetic_diversity_statistic,
)
from ete3 import Tree

TREE_DATA_DIR = Path(__file__).parent / "tree_data"


class TreeMetric(Protocol):
    def __call__(self, tree: Tree) -> float | int: ...


METRIC_FUNCTIONS: dict[str, TreeMetric] = {
    "cherries": cherry_count_statistic,
    "colless": colless_statistic,
    "phylogenetic_diversity": phylogenetic_diversity_statistic,
}


@pytest.fixture(scope="module")
def ground_truth_stats():
    csv_path = TREE_DATA_DIR / "tree_statistics.csv"
    df = pd.read_csv(csv_path)
    return df.set_index("tree_id")


@pytest.fixture(scope="module")
def newick_trees():
    trees = {}
    for tree_id in range(1, 11):
        tree_path = TREE_DATA_DIR / f"tree_{tree_id:02d}.newick"
        trees[tree_id] = Tree(str(tree_path), format=1)
    return trees


@pytest.mark.parametrize("tree_id", range(1, 11))
@pytest.mark.parametrize("metric", ["colless", "phylogenetic_diversity", "cherries"])
def test_tree_metric(tree_id, metric, ground_truth_stats, newick_trees):
    tree = newick_trees[tree_id]
    expected_value = ground_truth_stats.loc[tree_id, metric]

    if metric not in METRIC_FUNCTIONS:
        pytest.skip(f"Python implementation for {metric} not yet implemented")
        # raise NotImplementedError(f"Python implementation for {metric} not yet implemented")

    metric_fn = METRIC_FUNCTIONS[metric]
    actual_value = metric_fn(tree)
    assert actual_value == pytest.approx(expected_value)
