import subprocess
from pathlib import Path

import typer

app = typer.Typer(pretty_exceptions_enable=False)


@app.command()
def run_fasttree(alignment_file: Path, output_file: Path) -> None:
    with open(output_file, "w") as f:
        subprocess.run(["FastTree", str(alignment_file)], stdout=f, check=True)


if __name__ == "__main__":
    app()
