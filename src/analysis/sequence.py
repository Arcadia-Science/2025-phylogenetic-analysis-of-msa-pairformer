import subprocess
from pathlib import Path

import biotite.sequence as seq
import biotite.sequence.io.fasta as fasta


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
