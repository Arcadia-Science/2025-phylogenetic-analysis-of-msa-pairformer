import subprocess
import zipfile
from dataclasses import dataclass
from functools import partial
from pathlib import Path

from rich.console import Console
from tqdm import tqdm

_fmt = "{desc}: {percentage:3.0f}%|{bar}| {n_fmt}/{total_fmt} [{elapsed}]"
progress = partial(tqdm, bar_format=_fmt)

console = Console()


@dataclass
class ZenodoArtifact:
    """A data artifact that can be fetched from Zenodo.

    The zip file is retained after extraction to indicate that the data was fetched
    rather than computed de novo. To force recomputation, delete the extracted files
    and set fetch=False.
    """

    url: str
    extract_dir: Path
    fetch: bool = True

    @property
    def zip_path(self) -> Path:
        return self.extract_dir / Path(self.url).name

    def download_if_missing(self) -> None:
        self.extract_dir.mkdir(parents=True, exist_ok=True)

        if not self.fetch:
            return

        if self.zip_path.exists():
            return

        subprocess.run(
            ["wget", "-O", str(self.zip_path), self.url],
            check=True,
        )

        with zipfile.ZipFile(self.zip_path, "r") as zf:
            zf.extractall(self.extract_dir)
