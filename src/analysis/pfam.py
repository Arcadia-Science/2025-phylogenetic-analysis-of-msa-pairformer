import gzip
import shutil
import subprocess
import tempfile
from collections.abc import Callable
from pathlib import Path
from typing import Annotated, Any

import pandas as pd
import requests
import typer
from Bio import AlignIO
from Bio.Align import MultipleSeqAlignment

from analysis.sequence import reformat_alignment, unalign_fasta

app = typer.Typer(pretty_exceptions_enable=False)

INTERPRO_BASE = (
    "https://www.ebi.ac.uk/interpro/wwwapi/entry/pfam/{pfam}/?annotation=alignment:{atype}&download"
)


def download_pfam(pfam_id: str, sto_output_path: Path, atype: str = "seed") -> None:
    """Download and process a Pfam alignment from InterPro.

    Downloads a gzipped Stockholm alignment file from InterPro, extracts it to the
    specified path, and creates both aligned and unaligned FASTA versions. The
    function creates three output files: the Stockholm alignment, an aligned FASTA,
    and an unaligned FASTA with gap characters removed.

    Args:
        pfam_id: The Pfam identifier (e.g., "PF00072").
        sto_output_path: Path where the Stockholm alignment will be written.
        atype: Alignment type to download, either "full" or "seed".
    """
    url = _build_url(pfam_id, atype)

    headers = {
        "User-Agent": "PFAM downloader for MSA Pairformer phylogenetic analysis",
    }

    response = requests.get(url, headers=headers, stream=True)
    response.raise_for_status()

    with tempfile.NamedTemporaryFile(suffix=".gz") as temp_gz:
        for chunk in response.iter_content(chunk_size=1024 * 1024):
            temp_gz.write(chunk)
        temp_gz.flush()

        print(f"Downloaded {pfam_id}")
        _unzip(Path(temp_gz.name), sto_output_path)

    fasta_output_path = sto_output_path.with_suffix(".fasta")
    reformat_alignment(sto_output_path, fasta_output_path, "sto", "fas")

    unaligned_fasta_output_path = fasta_output_path.with_name(
        f"{fasta_output_path.stem}_unaligned.fasta"
    )
    unalign_fasta(fasta_output_path, unaligned_fasta_output_path)


@app.command()
def download_and_process_response_regulator_msa(
    output_dir: Annotated[
        Path,
        typer.Option(help="Artifacts will be stored in this directory."),
    ],
    subset_size: Annotated[
        int, typer.Option(help="Target number of sequences for diversity filtering (approximate)")
    ] = 4096,
):
    pfam_to_pdb_rep = {
        "PF00486": "1NXS",
        "PF04397": "4CBV",
        "PF00196": "4E7P",
    }
    subfamily_names = ["OmpR", "LytTR", "GerE"]
    subfamily_ids = list(pfam_to_pdb_rep.keys())
    references = list(pfam_to_pdb_rep.values())

    output_dir.mkdir(exist_ok=True)

    family_id = "PF00072"
    all_pfam_ids = [family_id] + subfamily_ids

    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)

        # -- Download all PFAMs needed
        for pfam_id in all_pfam_ids:
            sto_file = tmpdir / f"{pfam_id}.full.sto"
            _get_or_compute(
                sto_file,
                output_dir,
                lambda pfam_id=pfam_id, sto_file=sto_file: download_pfam(pfam_id, sto_file, "full"),
            )

        # --- Filter family PFAM to only include members from each subfamily
        source_sto = tmpdir / f"{family_id}.full.sto"
        filtered_sto = tmpdir / f"{family_id}.filtered.sto"
        _get_or_compute(
            filtered_sto,
            output_dir,
            lambda: _filter_sto(
                source=source_sto,
                output=filtered_sto,
                keep_if_in=[tmpdir / f"{pfam_id}.full.sto" for pfam_id in subfamily_ids],
            ),
        )

        # --- Convert from STO to A3M (for hhfilter)
        filtered_a3m = filtered_sto.with_suffix(".a3m")
        _get_or_compute(
            filtered_a3m,
            output_dir,
            lambda: reformat_alignment(filtered_sto, filtered_a3m, "sto", "a3m"),
        )

        # --- Subset with hhfilter down to the subset size
        subset_a3m = tmpdir / f"{family_id}.subset.a3m"
        _get_or_compute(
            subset_a3m,
            output_dir,
            lambda: _apply_hhfilter_strategy(
                input_a3m=filtered_a3m,
                output_a3m=subset_a3m,
                target_size=subset_size - len(subfamily_ids),
            ),
        )

        # --- Convert from A3M to STO (for HMMER)
        subset_sto = subset_a3m.with_suffix(".sto")
        _get_or_compute(
            subset_sto,
            output_dir,
            lambda: reformat_alignment(subset_a3m, subset_sto, "a3m", "sto"),
        )

        # --- Build an HMM profile with DESC line
        hmm_file = subset_sto.with_suffix(".hmm")
        _get_or_compute(
            hmm_file,
            output_dir,
            lambda: _build_hmm(
                hmm_file=hmm_file,
                sto_file=subset_sto,
                desc=f"{family_id} HMM Profile",
            ),
        )

        # --- Add in the representative sequences, saving aligned version
        pdb_fasta = output_dir / "pdb_rr.fasta"
        aligned_sto = tmpdir / f"{family_id}.final.sto"
        _get_or_compute(
            aligned_sto,
            output_dir,
            lambda: _hmmalign(
                hmm_file=hmm_file,
                mapali_file=subset_sto,
                sequences_file=pdb_fasta,
                output_file=aligned_sto,
            ),
        )

        # --- Convert aligned file to other formats
        _convert_alignment_formats(aligned_sto, output_dir)

        # --- Calculate membership and store results in TSV
        membership_path = tmpdir / "membership.txt"
        source_alignment_paths = {
            name: Path(output_dir / f"{id}.full.sto")
            for name, id in zip(subfamily_names, subfamily_ids, strict=True)
        }
        query_mapping = dict(zip(pfam_to_pdb_rep.values(), subfamily_names, strict=True))
        _get_or_compute(
            membership_path,
            output_dir,
            lambda: _calculate_membership(
                aligned_sto, query_mapping, source_alignment_paths, membership_path
            ),
        )

        for reference in references:
            # --- Reorder alignment with reference first
            final_sto = tmpdir / f"{family_id}.final_{reference}.sto"
            _get_or_compute(
                final_sto,
                output_dir,
                lambda ref=reference, final=final_sto: _reorder_alignment_with_reference_first(
                    input_sto=aligned_sto,
                    output_sto=final,
                    reference_id=ref,
                ),
            )

            # --- Convert final alignment to other formats
            _convert_alignment_formats(final_sto, output_dir)

    print("Processing complete!")


def _get_or_compute(
    tmp_file: Path,
    output_dir: Path,
    compute_fn: Callable[[], Any],
) -> None:
    """Check if file exists in output_dir, copy if yes, compute if no."""
    output_file = output_dir / tmp_file.name
    if output_file.exists():
        shutil.copy(output_file, tmp_file)
        print(f"Using cached: {tmp_file.name}")
        return

    if compute_fn:
        compute_fn()

    shutil.copy(tmp_file, output_file)


def _build_url(pfam_id: str, atype: str) -> str:
    return INTERPRO_BASE.format(pfam=pfam_id, atype=atype)


def _unzip(gz_file: Path, output_file: Path) -> Path:
    """Extract a .gz file and return path to extracted file."""
    with gzip.open(gz_file, "rb") as f_in:
        with open(output_file, "wb") as f_out:
            f_out.write(f_in.read())

    print(f"Extracted {gz_file} -> {output_file}")
    return output_file


def _build_hmm(hmm_file: Path, sto_file: Path, desc: str) -> None:
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


def _hmmalign(
    hmm_file: Path,
    mapali_file: Path,
    sequences_file: Path,
    output_file: Path,
) -> None:
    """
    Run hmmalign to align sequences using HMM profile.

    Args:
        hmm_file: HMM profile file
        mapali_file: Map alignment file (--mapali)
        sequences_file: FASTA file with sequences to thread
        output_file: Output STO file
    """
    subprocess.run(
        [
            "hmmalign",
            "--trim",
            "--mapali",
            str(mapali_file),
            str(hmm_file),
            str(sequences_file),
        ],
        stdout=open(output_file, "w"),
        check=True,
    )


def _reorder_alignment_with_reference_first(
    input_sto: Path,
    output_sto: Path,
    reference_id: str,
) -> None:
    """
    Reorder alignment to place reference sequence first.

    Args:
        input_sto: Input Stockholm alignment file
        output_sto: Output Stockholm alignment file
        reference_id: ID of the reference sequence to move to top
    """
    aln = AlignIO.read(input_sto, "stockholm")

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
    with open(output_sto, "w") as handle:
        AlignIO.write(reordered_aln, handle, "stockholm")


def _convert_alignment_formats(
    sto_file: Path,
    output_dir: Path,
) -> tuple[Path, Path, Path]:
    """
    Convert STO alignment to A3M, FASTA, and unaligned FASTA formats.

    Args:
        sto_file: Source Stockholm alignment file

    Returns:
        Tuple of (a3m_file, fasta_file, unaligned_fasta_file) paths
    """
    a3m_file = sto_file.with_suffix(".a3m")
    _get_or_compute(
        a3m_file,
        output_dir,
        lambda: reformat_alignment(sto_file, a3m_file, "sto", "a3m"),
    )

    fasta_file = sto_file.with_suffix(".fasta")
    _get_or_compute(
        fasta_file,
        output_dir,
        lambda: reformat_alignment(sto_file, fasta_file, "sto", "fas"),
    )

    unaligned_fasta_file = sto_file.with_suffix(".unaligned.fasta")
    _get_or_compute(
        unaligned_fasta_file,
        output_dir,
        lambda: unalign_fasta(fasta_file, unaligned_fasta_file),
    )

    return a3m_file, fasta_file, unaligned_fasta_file


def _filter_sto(source: Path, output: Path, keep_if_in: list[Path]) -> None:
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


def _apply_hhfilter_strategy(input_a3m: Path, output_a3m: Path, target_size: int = 4096) -> None:
    """Filter with hhfilter

    Note: the target size is first approximated using the hhfilter's `-diff` parameter,
    which yields a number of sequences close to, but in excess of `target_size`. We
    filter from this approximate size to exactly `target_size` by truncating the MSA,
    however it would be more proper to do a greedy filter based on hamming distance to
    the query. For this reason, and because the original authors did not provide exact
    commands used for their response regulator analysis, we don't expect our MSAs to
    perfectly match those used in the original paper.
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
                "-diff",
                str(target_size),
                "-maxseq",
                "200000",
            ],
            check=True,
            capture_output=True,
            text=True,
        )

        with open(intermediate_a3m) as f:
            lines = f.readlines()

        n_seqs = sum(1 for i, line in enumerate(lines) if i % 2 == 0 and line.startswith(">"))
        print(f"HHFilter returned {n_seqs} sequences (target: {target_size})")

        if n_seqs <= target_size:
            shutil.copy(intermediate_a3m, output_a3m)
        else:
            # If we got more than target, truncate to first target_size sequences
            print(f"Truncating from {n_seqs} to {target_size} sequences")

            with open(output_a3m, "w") as f_out:
                seq_count = 0
                for i, line in enumerate(lines):
                    if i % 2 == 0 and line.startswith(">"):
                        if seq_count >= target_size:
                            break
                        seq_count += 1
                        f_out.write(line)
                        if i + 1 < len(lines):
                            f_out.write(lines[i + 1])

        print(f"Final MSA has {min(n_seqs, target_size)} sequences")


def _calculate_membership(
    alignment_path: Path,
    query_mapping: dict[str, str],
    source_alignment_paths: dict[str, Path],
    output_path: Path,
) -> None:
    subfamilies = tuple(source_alignment_paths.keys())

    aln = AlignIO.read(alignment_path, "stockholm")

    members = {}
    for subfamily, source_path in source_alignment_paths.items():
        ids = [record.id.split("/")[0] for record in AlignIO.read(source_path, "stockholm")]
        members[subfamily] = set(ids)

    membership = {}
    for record in aln:
        for subfamily in subfamilies:
            if record.id.split("/")[0] in members[subfamily]:
                membership[record.id] = subfamily
                break
        else:
            print(record.id)

    membership.update(query_mapping)

    (
        pd.Series(membership)
        .to_frame()
        .reset_index()
        .rename(columns={"index": "record_id", 0: "subfamily"})
        .to_csv(output_path, sep="\t", index=False)
    )


if __name__ == "__main__":
    app()
