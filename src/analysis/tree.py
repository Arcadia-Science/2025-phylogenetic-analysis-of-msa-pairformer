import random
import subprocess
from pathlib import Path

import pandas as pd
import typer
from ete3 import Tree

app = typer.Typer(pretty_exceptions_enable=False)


def read_newick(path: str | Path) -> Tree:
    return Tree(str(path), format=1)


def get_patristic_distance(tree: Tree, reference: str) -> pd.Series:
    for leaf in tree.get_leaves():
        if reference in leaf.name:
            reference_node = leaf
            break
    else:
        raise ValueError(f"Reference '{reference}' not found in tree leaves")

    return pd.Series({leaf.name: reference_node.get_distance(leaf) for leaf in tree.get_leaves()})


def subset_tree(tree: Tree, n: int, force_include: list[str] | None = None, seed=42) -> Tree:
    random.seed(seed)

    if force_include is None:
        force_include = []

    all_leaves = [leaf.name for leaf in tree.get_leaves()]

    if n > len(all_leaves):
        n = len(all_leaves)

    available_leaves = [leaf for leaf in all_leaves if leaf not in force_include]
    n_random = n - len(force_include)

    if n_random < 0:
        selected_leaves = force_include[:n]
    else:
        random_leaves = random.sample(available_leaves, min(n_random, len(available_leaves)))
        selected_leaves = force_include + random_leaves

    subset_tree = tree.copy()
    subset_tree.prune(selected_leaves)

    return subset_tree


@app.command()
def run_fasttree(alignment_file: Path, output_file: Path) -> None:
    with open(output_file, "w") as f:
        subprocess.run(["FastTree", str(alignment_file)], stdout=f, check=True)


if __name__ == "__main__":
    app()
