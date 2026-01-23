import argparse
import re
import subprocess
import tempfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pandas as pd
from src.analysis.sequence import reformat_alignment


def load_existing_results(output_file: Path) -> dict[str, str]:
    if not output_file.exists():
        return {}
    df = pd.read_csv(output_file)
    return {row["query"]: row["difficulty"] for _, row in df.iterrows()}


def save_results(results: dict[str, str], output_file: Path) -> None:
    with open(output_file, "w") as f:
        f.write("query,difficulty\n")
        for query in sorted(results.keys()):
            f.write(f"{query},{results[query]}\n")


def run_pythia(a3m_file: Path, raxml_path: Path) -> tuple[str, str]:
    query = a3m_file.stem

    with tempfile.TemporaryDirectory() as tmpdir:
        fasta_file = Path(tmpdir) / f"{query}.fasta"
        reformat_alignment(a3m_file, fasta_file, "a3m", "fas")

        cmd = ["pythia", "-m", str(fasta_file), "-r", str(raxml_path), "-t", "1"]
        result = subprocess.run(cmd, capture_output=True, text=True)
        output = result.stdout + result.stderr

    match = re.search(r"The predicted difficulty for MSA .+ is: ([\w.]+)", output)
    if match:
        return query, match.group(1)
    print(f"DEBUG {query} output:\n{output[:1000]}")
    return query, "PARSE_ERROR"


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Pythia difficulty prediction on A3M files")
    parser.add_argument("a3m_dir", type=Path, help="Directory containing A3M files")
    parser.add_argument("--output", type=Path, default=Path("msa_difficulty.csv"), help="Output CSV file")
    parser.add_argument("--raxml-path", type=Path, default=Path("/Users/evan/miniconda3/bin/raxml-ng"), help="Path to raxml-ng")
    parser.add_argument("--max-workers", type=int, default=12, help="Number of parallel workers")
    parser.add_argument("--save-interval", type=int, default=20, help="Save results every N completions")
    args = parser.parse_args()

    results = load_existing_results(args.output)
    print(f"Loaded {len(results)} existing results")

    a3m_files = [f for f in args.a3m_dir.glob("*.a3m") if f.stem not in results]
    total = len(a3m_files)
    print(f"Processing {total} remaining files")

    completed = 0

    with ThreadPoolExecutor(max_workers=args.max_workers) as executor:
        futures = {
            executor.submit(run_pythia, f, args.raxml_path): f
            for f in a3m_files
        }
        for future in as_completed(futures):
            query, difficulty = future.result()
            results[query] = difficulty
            completed += 1
            print(f"[{completed}/{total}] {query}: difficulty={difficulty}")

            if completed % args.save_interval == 0:
                save_results(results, args.output)
                print(f"  (saved {len(results)} results)")

    save_results(results, args.output)
    print(f"\nDone. Wrote {len(results)} total results to {args.output}")


if __name__ == "__main__":
    main()
