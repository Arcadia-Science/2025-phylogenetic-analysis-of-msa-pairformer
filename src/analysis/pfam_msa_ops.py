import gzip
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Annotated

import requests
import typer
from Bio import AlignIO
from Bio.Align import MultipleSeqAlignment
from Bio.PDB.PDBList import PDBList
from Bio.PDB.PDBParser import PDBParser
from Bio.PDB.Polypeptide import PPBuilder

INTERPRO_BASE = (
    "https://www.ebi.ac.uk/interpro/wwwapi/entry/pfam/{pfam}/?annotation=alignment:{atype}&download"
)

OUTDIR = Path("pfam_alignments")


def build_url(pfam_id: str, atype: str) -> str:
    return INTERPRO_BASE.format(pfam=pfam_id, atype=atype)


def download_pfam(pfam_id: str, output_dir: Path, atype: str = "full") -> None:
    url = build_url(pfam_id, atype)
    filename = f"{pfam_id}.alignment.{atype}.gz"
    dest = output_dir / filename

    headers = {
        "User-Agent": "PFAM downloader for MSA Pairformer phylogenetic analysis",
    }

    response = requests.get(url, headers=headers, stream=True)
    response.raise_for_status()

    with open(dest, "wb") as f:
        for chunk in response.iter_content(chunk_size=1024 * 1024):
            f.write(chunk)

    print(f"Downloaded {pfam_id} -> {dest}")


def download_pfams(
    pfam_ids: list[str],
    output_dir: Path = OUTDIR,
):
    output_dir.mkdir(exist_ok=True)
    for pfam_id in pfam_ids:
        download_pfam(pfam_id, output_dir)

    print("All downloads completed.")


def unzip(gz_file: Path, output_file: Path) -> Path:
    """Extract a .gz file and return path to extracted file."""
    with gzip.open(gz_file, "rb") as f_in:
        with open(output_file, "wb") as f_out:
            f_out.write(f_in.read())

    print(f"Extracted {gz_file} -> {output_file}")
    return output_file


def build_hmm(hmm_file: Path, sto_file: Path, desc: str) -> None:
    """Build HMM and add DESC line to it."""
    subprocess.run(
        [
            "hmmbuild",
            str(hmm_file),
            str(sto_file),
        ],
        check=True,
    )

    # hmmalign wants a non-empty DESC line, so mutate line to include one.
    with open(hmm_file) as f:
        lines = f.readlines()

    with open(hmm_file, "w") as f:
        for line in lines:
            if line.startswith("DESC  "):
                f.write(f"DESC  {desc}\n")
            else:
                f.write(line)


def hmmalign_with_reference_first(
    hmm_file: Path,
    mapali_file: Path,
    sequences_file: Path,
    output_file: Path,
    reference_id: str,
) -> None:
    """
    Run hmmalign and ensure reference sequence appears first in the output STO file.

    Args:
        hmm_file: HMM profile file
        mapali_file: Map alignment file (--mapali)
        sequences_file: FASTA file with sequences to thread
        output_file: Output STO file
        reference_id: ID of the reference sequence to move to top
    """
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_sto = Path(temp_dir) / "temp_alignment.sto"

        subprocess.run(
            [
                "hmmalign",
                "--trim",
                "--mapali",
                str(mapali_file),
                str(hmm_file),
                str(sequences_file),
            ],
            stdout=open(temp_sto, "w"),
            check=True,
        )

        aln = AlignIO.read(temp_sto, "stockholm")

        # Find reference sequence and move it to front
        reference_record = None
        other_records = []

        for record in aln:
            if reference_id in record.id:
                reference_record = record
            else:
                other_records.append(record)

        if reference_record is None:
            raise ValueError(f"Reference sequence '{reference_id}' not found in alignment")

        reordered_aln = MultipleSeqAlignment([reference_record] + other_records)
        with open(output_file, "w") as handle:
            AlignIO.write(reordered_aln, handle, "stockholm")


def filter_sto(source: Path, output: Path, keep_if_in: list[Path]) -> None:
    keep = set()
    for pfam_aln_path in keep_if_in:
        aln = AlignIO.read(pfam_aln_path, "stockholm")
        for record in aln:
            keep.add(record.id.split("/")[0])

    filtered_aln = MultipleSeqAlignment(records=[])
    full_aln = AlignIO.read(source, "stockholm")
    for record in full_aln:
        if record.id.split("/")[0] in keep:
            filtered_aln._records.append(record)

    print(f"Filtered from {len(full_aln)} to {len(filtered_aln)} sequences.")

    with open(output, "w") as handle:
        AlignIO.write(filtered_aln, handle, "stockholm")

    print(f"Wrote alignment to {output}.")


def apply_hhfilter_strategy(input_a3m: Path, output_a3m: Path, target_size: int = 4096) -> None:
    """
    Apply HHFilter strategy as described in the paper:
    0. "Removing sequences with less than 50% coverage, less than 15% sequence identity
       to the query, and setting a maximum pairwise sequence identity of 95%, a
       minimum."
    1. Use hhfilter with -diff parameter to maximize diversity
    2. If more sequences than target are returned, truncate to target size
    """
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_dir = Path(temp_dir)
        intermediate_a3m = temp_dir / "hhfilter_intermediate.a3m"

        subprocess.run(
            [
                "hhfilter",
                "-i",
                str(input_a3m),
                "-o",
                str(intermediate_a3m),
                # "-cov",
                # "50",
                # "-qid",
                # "15",
                # "-id",
                # "95",
                "-diff",
                str(target_size),
                "-maxseq",
                "200000",
            ],
            check=True,
            capture_output=True,
            text=True,
        )

        # Count sequences in the output (a3m has one defline + one sequence per entry)
        with open(intermediate_a3m) as f:
            lines = f.readlines()

        # Count sequences (every other line starting from 0 is a defline)
        n_seqs = sum(1 for i, line in enumerate(lines) if i % 2 == 0 and line.startswith(">"))
        print(f"HHFilter returned {n_seqs} sequences (target: {target_size})")

        if n_seqs <= target_size:
            # If we got target size or fewer, just use the result
            shutil.copy(intermediate_a3m, output_a3m)
        else:
            # If we got more than target, truncate to first target_size sequences
            # Since hhfilter already maximized diversity, the order should be good
            print(f"Truncating from {n_seqs} to {target_size} sequences")

            with open(output_a3m, "w") as f_out:
                seq_count = 0
                for i, line in enumerate(lines):
                    if i % 2 == 0 and line.startswith(">"):
                        if seq_count >= target_size:
                            break
                        seq_count += 1
                        f_out.write(line)
                        # Write the sequence line (next line)
                        if i + 1 < len(lines):
                            f_out.write(lines[i + 1])

        print(f"Final MSA has {min(n_seqs, target_size)} sequences")


def fetch_pdb_rr_domains(output_file: Path, pdb_entries: dict[str, str]) -> None:
    pdbl = PDBList()
    parser = PDBParser(QUIET=True)
    ppb = PPBuilder()

    sequences = []

    with tempfile.TemporaryDirectory() as tmpdir:
        for pdb_id in pdb_entries.values():
            pdb_file = pdbl.retrieve_pdb_file(pdb_id, pdir=tmpdir, file_format="pdb")
            structure = parser.get_structure(pdb_id, pdb_file)

            assert structure is not None
            for model in structure:
                if "A" in model:
                    chain = model["A"]
                    for pp in ppb.build_peptides(chain):
                        seq = pp.get_sequence()
                        sequences.append((f"{pdb_id}", str(seq)))
                        break
                    break

    with open(output_file, "w") as f:
        for name, seq in sequences:
            f.write(f">{name}\n{seq}\n")

    print(f"Saved {len(sequences)} PDB sequences to {output_file}")


def process(
    family_id: Annotated[
        str,
        typer.Option(help="Main family ID ({family_id} in paper)"),
    ],
    subfamily_ids: Annotated[
        list[str],
        typer.Option(help="Subfamily IDs ( in paper)"),
    ],
    subset_size: Annotated[
        int, typer.Option(help="Target number of sequences for diversity filtering (approximate)")
    ] = 4096,
    reference: Annotated[
        str,
        typer.Option(help="The sequence treated as the query (default: 4CBV)"),
    ] = "4CBV",
    output_dir: Annotated[
        Path,
        typer.Option(help="MSA artefacts will be stored in this directory."),
    ] = OUTDIR,
    keep_intermediate: Annotated[
        bool,
        typer.Option(help="Whether to keep intermediate files"),
    ] = False,
):
    output_dir.mkdir(exist_ok=True)

    # TODO: This mapping is currently hard-coded.
    pfam_to_pdb_rep = {
        "PF00486": "1NXS",
        "PF04397": "4CBV",
        "PF00196": "4E7P",
    }
    assert reference in pfam_to_pdb_rep.values()

    all_pfam_ids = [family_id] + subfamily_ids

    def _get_or_compute(tmp_file: Path, compute_fn=None) -> None:
        """Check if file exists in output_dir, copy if yes, compute if no."""
        output_file = output_dir / tmp_file.name
        if output_file.exists():
            shutil.copy(output_file, tmp_file)
            print(f"Using cached: {tmp_file.name}")
            return

        if compute_fn:
            compute_fn()

        if keep_intermediate:
            shutil.copy(tmp_file, output_file)

    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)

        # -- Download all PFAMs needed
        for pfam_id in all_pfam_ids:
            gz_file = output_dir / f"{pfam_id}.alignment.full.gz"
            if not gz_file.exists():
                print(f"Downloading {pfam_id}...")
                download_pfam(pfam_id, output_dir)

            sto_file = tmpdir / f"{pfam_id}.alignment.full.sto"
            _get_or_compute(sto_file, lambda gz=gz_file, sto=sto_file: unzip(gz, sto))

        # --- Subset family PFAM to only include members from each subfamily
        source_sto = tmpdir / f"{family_id}.alignment.full.sto"
        filtered_sto = tmpdir / f"{family_id}.alignment.filtered.sto"
        _get_or_compute(
            filtered_sto,
            lambda: filter_sto(
                source=source_sto,
                output=filtered_sto,
                keep_if_in=[tmpdir / f"{pfam_id}.alignment.full.sto" for pfam_id in subfamily_ids],
            ),
        )

        # --- Create a FASTA of representatives (TODO currently hard-coded representatives)
        pdb_fasta = tmpdir / "pdb_rr.fasta"
        _get_or_compute(
            pdb_fasta,
            lambda: fetch_pdb_rr_domains(
                pdb_fasta,
                pfam_to_pdb_rep,
            ),
        )

        # --- Build an HMM profile with DESC line
        hmm_file = filtered_sto.with_suffix(".hmm")
        _get_or_compute(
            hmm_file,
            lambda: build_hmm(
                hmm_file=hmm_file,
                sto_file=filtered_sto,
                desc=f"{family_id} HMM Profile",
            ),
        )

        # --- Add in the representative, saving to STO
        filtered_with_ref_sto = tmpdir / f"{family_id}.alignment.filtered_with_{reference}.sto"
        _get_or_compute(
            filtered_with_ref_sto,
            lambda: hmmalign_with_reference_first(
                hmm_file=hmm_file,
                mapali_file=filtered_sto,
                sequences_file=pdb_fasta,
                output_file=filtered_with_ref_sto,
                reference_id=reference,
            ),
        )

        # --- Convert from STO to A3M (for hhblits filtering)
        filtered_with_ref_a3m = filtered_with_ref_sto.with_suffix(".a3m")
        _get_or_compute(
            filtered_with_ref_a3m,
            lambda: subprocess.run(
                [
                    "reformat.pl",
                    "sto",
                    "a3m",
                    str(filtered_with_ref_sto),
                    str(filtered_with_ref_a3m),
                ],
                check=True,
            ),
        )

        # --- Filter with hhfilter down to the subset size
        subset_a3m = tmpdir / f"{family_id}.alignment.final_{reference}.a3m"
        _get_or_compute(
            subset_a3m,
            lambda: apply_hhfilter_strategy(
                input_a3m=filtered_with_ref_a3m,
                output_a3m=subset_a3m,
                target_size=subset_size,
            ),
        )

        # --- Convert to STO for convenient reference
        subset_sto = subset_a3m.with_suffix(".sto")
        _get_or_compute(
            subset_sto,
            lambda: subprocess.run(
                [
                    "reformat.pl",
                    "a3m",
                    "sto",
                    str(subset_a3m),
                    str(subset_sto),
                ],
                check=True,
            ),
        )

        if not keep_intermediate:
            # Copy only the final subset MSA to output dir
            output_a3m = output_dir / subset_a3m.name
            if not output_a3m.exists():
                shutil.copy(subset_a3m, output_a3m)

            output_sto = output_dir / subset_sto.name
            if not output_sto.exists():
                shutil.copy(subset_sto, output_sto)

    print("Processing complete!")


if __name__ == "__main__":
    typer.run(process)
