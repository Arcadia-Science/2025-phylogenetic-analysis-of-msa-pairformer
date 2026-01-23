import shutil
import subprocess
import tempfile
from collections.abc import Sequence
from pathlib import Path
from typing import Literal

import biotite.sequence as seq
import biotite.sequence.io.fasta as fasta
from ete3 import Tree
from MSA_Pairformer.MSA_Pairformer.dataset import MSA


def unalign_fasta(aligned_fasta: Path, unaligned_fasta: Path) -> None:
    """Remove gaps from aligned FASTA sequences with biotite's elegance."""
    alignment = fasta.FastaFile.read(aligned_fasta)

    unaligned_sequences = {
        header: seq.ProteinSequence("".join(sequence.split("-")))
        for header, sequence in alignment.items()
    }

    unaligned_file = fasta.FastaFile()
    for header, sequence in unaligned_sequences.items():
        unaligned_file[header] = str(sequence)

    unaligned_file.write(unaligned_fasta)


def reformat_alignment(
    input_file: Path, output_file: Path, input_format: str, output_format: str
) -> None:
    """Convert between alignment formats using reformat.pl"""
    subprocess.run(
        [
            "reformat.pl",
            input_format,
            output_format,
            str(input_file),
            str(output_file),
        ],
        check=True,
    )


def write_fasta_like(seqs: Sequence[str], deflines: Sequence[str], output: Path) -> None:
    """Write a FASTA-like file.

    This performs no transformations on the sequence strings, which allows for FASTA
    extensions like A3M and A2M to be written, assuming the passed sequence strings
    match the format.
    """
    with open(output, "w") as fp:
        for seq, defline in zip(seqs, deflines, strict=True):
            fp.write(f">{defline}\n")
            fp.write(f"{seq}\n")


def filter_msa_by_tree(msa: MSA, tree: Tree) -> tuple[list[str], list[str]]:
    """Filters MSA by names matching tree leaf names.

    Returns:
        tuple[list[str], list[str]]:
            First list is sequence strings in A3M format, second list is deflines.

    Notes:
        - It would be natural to return a new `MSA_Pairformer.dataset.MSA` object, but
          the constructor for this class must load an A3M path, making construction of
          this object *without* an A3M difficult.
    """
    tree_leaf_ids = set(tree.get_leaf_names())

    # Get all sequences, maintaining insertions.
    unfiltered_seqs, unfiltered_ids = msa.parse_a3m_file(
        keep_insertions=True,
        to_upper=False,
        remove_lowercase_cols=False,
    )

    filtered_ids = []
    filtered_seqs = []

    for idx in msa.select_diverse_indices:
        unfiltered_id = unfiltered_ids[idx]
        unfiltered_seq = unfiltered_seqs[idx]

        if unfiltered_id in tree_leaf_ids:
            filtered_ids.append(unfiltered_id)
            filtered_seqs.append(unfiltered_seq)

    return filtered_seqs, filtered_ids


def write_processed_msa(
    msa: MSA,
    output: Path,
    format: Literal["fasta", "a3m", "unaligned_fasta"],
    simplify_ids: bool = False,
) -> None:
    """Write an MSA processed by MSA Pairformer to file.

    This is needed because `MSA_Pairformer.dataset.MSA` filters the number of sequences
    in the raw MSAs in the OpenProteinSet, and it is this filtered MSA we're interested
    in performing downstream calculations on, like calculating a tree. No function in
    the MSA_Pairformer codebase exists for this.

    Args:
        simplify_ids:
            If True, will convert deflines following the OpenProteinSet convention
            (e.g., `tr|A0A0N5AJP5|A0A0N5AJP5_9BILA`) to a simplified form (e.g.,
            `A0A0N5AJP5`)
    """

    # We keep all insertions, since these are meaningful for tree-building and other
    # potential downstream applications.
    unfiltered_seqs, unfiltered_ids = msa.parse_a3m_file(
        keep_insertions=True,
        to_upper=False,
        remove_lowercase_cols=False,
    )

    filtered_seqs = []
    filtered_ids = []
    for idx in msa.select_diverse_indices:
        filtered_seqs.append(unfiltered_seqs[idx])

        if simplify_ids:
            filtered_ids.append(unfiltered_ids[idx].split("|")[1])
        else:
            filtered_ids.append(unfiltered_ids[idx])

    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)

        a3m_path = tmpdir / "msa.a3m"
        write_fasta_like(filtered_seqs, filtered_ids, a3m_path)

        if format == "a3m":
            shutil.move(a3m_path, output)
            return

        fasta_path = tmpdir / "msa.fasta"
        reformat_alignment(a3m_path, fasta_path, "a3m", "fas")

        if format == "fasta":
            shutil.move(fasta_path, output)
            return

        unaligned_fasta_path = output
        unalign_fasta(fasta_path, unaligned_fasta_path)

        if format == "unaligned_fasta":
            shutil.move(unaligned_fasta_path, output)
            return

    raise ValueError(f"Unrecognized format: '{format}'")
