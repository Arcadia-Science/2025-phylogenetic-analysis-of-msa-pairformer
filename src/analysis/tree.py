import asyncio
import random
import subprocess
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
from ete3 import Tree
from scipy import stats


def run_fasttree(alignment_file: Path, output_file: Path, quiet: bool = False) -> None:
    stderr = subprocess.DEVNULL if quiet else None
    with open(output_file, "w") as f:
        subprocess.run(["FastTree", str(alignment_file)], stdout=f, stderr=stderr, check=True)


async def run_fasttree_async(
    input_a3m: Path,
    output_newick: Path,
    log_file: Path,
    semaphore: asyncio.Semaphore,
) -> None:
    """Runs FastTree asynchronously.

    Converts a3m to fasta in a temporary directory before running FastTree.
    """
    async with semaphore:
        with tempfile.TemporaryDirectory() as tmpdir:
            fasta_path = Path(tmpdir) / "aln.fasta"

            reformat_process = await asyncio.create_subprocess_exec(
                "reformat.pl",
                "a3m",
                "fas",
                str(input_a3m),
                str(fasta_path),
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            await reformat_process.wait()

            with open(output_newick, "w") as output_pointer, open(log_file, "w") as log_pointer:
                fasttree_process = await asyncio.create_subprocess_exec(
                    "FastTree",
                    str(fasta_path),
                    stdout=output_pointer,
                    stderr=log_pointer,
                )
                await fasttree_process.wait()


def read_newick(path: str | Path) -> Tree:
    return Tree(str(path), format=1)


def write_newick(tree: Tree, path: str | Path) -> None:
    tree.write(outfile=str(path), format=1)


def get_patristic_distance(tree: Tree, reference: str) -> pd.Series:
    for leaf in tree.get_leaves():
        if reference in leaf.name:
            reference_node = leaf
            break
    else:
        raise ValueError(f"Reference '{reference}' not found in tree leaves")

    return pd.Series({leaf.name: reference_node.get_distance(leaf) for leaf in tree.get_leaves()})


def build_distance_matrix(tree: Tree) -> np.ndarray:
    """Build pairwise distance matrix for all nodes in the tree.

    Args:
        tree: The phylogenetic tree

    Returns:
        Distance matrix where D[i,j] is the patristic distance between nodes i and j
    """
    all_nodes = list(tree.traverse("postorder"))
    n_nodes = len(all_nodes)

    dist_matrix = np.zeros((n_nodes, n_nodes))

    for i, node_a in enumerate(all_nodes):
        for j, node_b in enumerate(all_nodes):
            if i != j:
                dist_matrix[i, j] = node_a.get_distance(node_b)

    return dist_matrix


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


def subset_tree_around_reference(
    tree: Tree,
    n: int,
    reference: str,
    bias_power: float = 1.0,
    seed: int = 42,
) -> Tree:
    """Subset a tree by biasedly sampling leaves around a reference node.

    This function samples leaves with preference for those phylogenetically close to the
    reference, using inverse rank-based weighting. The reference node is always included.

    Args:
        tree: The phylogenetic tree to subset
        n: Total number of leaves to include in the subset (including reference)
        reference: Identifier substring to match the reference leaf node
        bias_power: Exponent to control sampling bias strength. Higher values (>1.0) increase
            bias toward the reference, lower values (<1.0) flatten the distribution for more
            uniform sampling. Default 1.0 uses inverse rank weighting.
        seed: Random seed for reproducible sampling

    Returns:
        Tree: A pruned copy of the input tree containing n leaves, with the reference
            and n-1 other leaves sampled with bias toward phylogenetic proximity to
            the reference
    """
    random.seed(seed)

    all_leaves = [leaf.name for leaf in tree.get_leaves()]

    if n > len(all_leaves):
        n = len(all_leaves)

    distances = get_patristic_distance(tree, reference)

    sorted_leaves = distances.sort_values().index.tolist()

    reference_idx = next(i for i, name in enumerate(sorted_leaves) if reference in name)
    sorted_leaves_without_ref = [name for name in sorted_leaves if reference not in name]

    weights = []
    for _, name in enumerate(sorted_leaves_without_ref):
        original_idx = sorted_leaves.index(name)
        distance_from_ref = abs(original_idx - reference_idx)
        weight = (1.0 / (distance_from_ref + 1)) ** bias_power
        weights.append(weight)

    total_weight = sum(weights)
    probabilities = [w / total_weight for w in weights]

    n_to_sample = min(n - 1, len(sorted_leaves_without_ref))

    rng = np.random.default_rng(seed)
    selected_indices = rng.choice(
        len(sorted_leaves_without_ref), size=n_to_sample, replace=False, p=probabilities
    )
    selected_leaves = [sorted_leaves_without_ref[i] for i in selected_indices]

    reference_leaf = next(name for name in sorted_leaves if reference in name)
    selected_leaves = [reference_leaf] + selected_leaves

    subset_tree_result = tree.copy()
    subset_tree_result.prune(selected_leaves)

    return subset_tree_result


def sort_tree_by_reference(tree: Tree, reference: str) -> Tree:
    """Sort tree topology so leaves roughly read from low to high patristic distance.

    This function reorders children at each internal node based on the minimum patristic
    distance to the reference among their descendants. This creates a visual ordering
    where leaves tend to appear from closest to farthest from the reference.

    Args:
        tree: The phylogenetic tree to sort
        reference: Identifier substring to match the reference leaf node

    Returns:
        Tree: A copy of the tree with reordered topology (branch lengths unchanged)
    """
    distances = get_patristic_distance(tree, reference)
    distance_dict = distances.to_dict()

    sorted_tree = tree.copy()

    def get_min_distance(node):
        if node.is_leaf():
            return distance_dict.get(node.name, float("inf"))
        return min(get_min_distance(child) for child in node.children)

    for node in sorted_tree.traverse("postorder"):
        if not node.is_leaf():
            node.children = sorted(node.children, key=get_min_distance)

    return sorted_tree


# --- Global tree statistics


def get_cherries(tree: Tree) -> list[tuple[str, str]]:
    """Find all cherries in the tree.

    A cherry is a pair of leaves that share an immediate common ancestor.

    Args:
        tree: The phylogenetic tree to analyze

    Returns:
        list[tuple[str, str]]: List of cherry pairs as tuples of leaf names
    """
    cherries = []
    for node in tree.traverse():
        if not node.is_leaf() and len(node.children) == 2:
            child1, child2 = node.children
            if child1.is_leaf() and child2.is_leaf():
                cherries.append((child1.name, child2.name))
    return cherries


def cherry_count_statistic(tree: Tree) -> int:
    return len(get_cherries(tree))


def colless_statistic(tree: Tree) -> int:
    stat = 0
    for node in tree.traverse("postorder"):
        if not node.is_leaf() and len(node.children) == 2:
            left, right = node.children
            left_tips = len(left.get_leaves())
            right_tips = len(right.get_leaves())
            stat += abs(left_tips - right_tips)
    return stat


def phylogenetic_diversity_statistic(tree: Tree) -> float:
    total = 0.0
    for node in tree.traverse():
        if node.dist:
            total += node.dist
    return total


def distance_laplacian_spectrum(tree: Tree) -> dict[str, np.ndarray | float]:
    """Calculate distance Laplacian spectrum statistics.

    Computes eigenvalues of the distance Laplacian matrix and returns the principal
    information needed for statistics 41-44 from Lewitus & Morlon (2016):
    - Principal eigenvalue
    - Eigengap (position of largest difference between eigenvalues)
    - Asymmetry (skewness of eigenvalue distribution)
    - Peakedness (max of spectral density)

    Args:
        tree: The phylogenetic tree

    Returns:
        Dictionary with keys:
            - eigenvalues: All eigenvalues >= 1
            - principal_eigenvalue: Maximum eigenvalue
            - eigengap: Position of largest gap between consecutive eigenvalues
            - asymmetry: Skewness of eigenvalue distribution
            - peakedness: Maximum of the spectral density distribution
    """
    dist_matrix = build_distance_matrix(tree)

    row_sums = dist_matrix.sum(axis=1)
    distance_laplacian = np.diag(row_sums) - dist_matrix

    eigenvalues = np.linalg.eigvalsh(distance_laplacian)
    eigenvalues = np.sort(eigenvalues[eigenvalues >= 1])[::-1]

    principal_eigenvalue = eigenvalues[0]

    gaps = np.abs(np.diff(eigenvalues))
    eigengap_position = int(np.argmax(gaps) + 1)  # 1-indexed to match treestats

    asymmetry = stats.skew(eigenvalues)

    log_eigenvalues = np.log(eigenvalues)
    std = np.std(log_eigenvalues, ddof=1)
    iqr = np.percentile(log_eigenvalues, 75) - np.percentile(log_eigenvalues, 25)

    # This matches R's built-in function `stats::bw.nrd0`: for bandwidth estimation.
    # https://stat.ethz.ch/R-manual/R-patched/library/stats/html/bandwidth.html
    bw = 0.9 * min(std, iqr / 1.34) * len(log_eigenvalues) ** (-1 / 5)

    x_range = np.linspace(log_eigenvalues.min() - 3 * bw, log_eigenvalues.max() + 3 * bw, 4096)

    density = np.zeros_like(x_range)
    for log_eig in log_eigenvalues:
        density += stats.norm.pdf(x_range, loc=log_eig, scale=bw)
    density /= len(log_eigenvalues)

    integral = np.trapz(density, x_range)
    density_normalized = density / integral

    peakedness = density_normalized.max()

    return {
        "eigenvalues": eigenvalues,
        "principal_eigenvalue": principal_eigenvalue,
        "eigengap": eigengap_position,
        "asymmetry": asymmetry,
        "peakedness": peakedness,
    }


def _root_to_tip_distances(tree: Tree) -> np.ndarray:
    return np.array([tree.get_distance(leaf) for leaf in tree.get_leaves()])


def ultrametricity_cv(tree: Tree) -> float:
    """Calculate coefficient of variation of root-to-tip distances.

    Perfect ultrametric trees have CV = 0. This is a scale-free measure.

    Args:
        tree: The phylogenetic tree

    Returns:
        Coefficient of variation (std / mean) of root-to-tip distances
    """
    distances = _root_to_tip_distances(tree)
    mean_dist = np.mean(distances)
    std_dist = np.std(distances, ddof=1)
    return float(std_dist / mean_dist)


# --- Query-centric metrics


def patristic_std(tree: Tree, reference: str) -> float:
    distances = get_patristic_distance(tree, reference).values
    return distances.std()


def query_centrality(tree: Tree, reference: str) -> float:
    """Calculate normalized centrality of query in tree topology.

    This metric measures whether the query is in the center or periphery of the
    tree by comparing its mean distance to all other leaves against the overall
    mean pairwise distance. Values < 1 indicate central position, > 1 indicate
    peripheral position.

    Args:
        tree: The phylogenetic tree
        reference: Identifier substring to match the reference leaf node

    Returns:
        Ratio of mean query distance to mean pairwise distance
    """
    distances = get_patristic_distance(tree, reference)
    mean_query_distance = distances.mean()

    all_leaves = list(tree.get_leaves())
    pairwise_distances = []

    for i, leaf1 in enumerate(all_leaves):
        for leaf2 in all_leaves[i + 1 :]:
            pairwise_distances.append(leaf1.get_distance(leaf2))

    mean_pairwise_distance = np.mean(pairwise_distances)

    return mean_query_distance / mean_pairwise_distance
