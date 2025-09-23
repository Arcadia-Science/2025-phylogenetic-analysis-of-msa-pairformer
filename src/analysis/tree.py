import subprocess
from pathlib import Path

import networkx as nx
import pandas as pd
import typer
from Bio import Phylo
from Bio.Phylo.Newick import Tree

app = typer.Typer(pretty_exceptions_enable=False)


def read_newick(path: Path) -> Tree:
    return Phylo.read(path, "newick")  # type: ignore


def get_patristic_distance(tree: Tree, reference: str) -> pd.Series:
    nx_tree = Phylo.to_networkx(tree)  # type: ignore

    reference_leaf = None

    leaves = []
    for node in nx_tree.nodes():
        if nx_tree.degree(node) == 1:
            leaves.append(node)

            if node.name == reference:
                reference_leaf = node

    assert reference_leaf is not None

    distances = {}
    for leaf in leaves:
        distances[leaf.name] = nx.shortest_path_length(
            nx_tree,
            reference_leaf,
            leaf,
            weight="weight",
        )

    return pd.Series(distances)


@app.command()
def run_fasttree(alignment_file: Path, output_file: Path) -> None:
    with open(output_file, "w") as f:
        subprocess.run(["FastTree", str(alignment_file)], stdout=f, check=True)


if __name__ == "__main__":
    app()
