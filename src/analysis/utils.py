import subprocess
import zipfile
from dataclasses import dataclass
from functools import partial
from pathlib import Path

import boto3
from botocore import UNSIGNED
from botocore.config import Config
from botocore.exceptions import ClientError
from rich.console import Console
from tqdm import tqdm

_fmt = "{desc}: {percentage:3.0f}%|{bar}| {n_fmt}/{total_fmt} [{elapsed}]"
progress = partial(tqdm, bar_format=_fmt)

console = Console()


PUBLIC_S3_BUCKET = "2025-phylogenetic-analysis-of-msa-pairformer"
ZENODO_BASE_URL = "https://zenodo.org/records/18318397/files"


def _download_from_s3(bucket: str, key: str, dest: Path) -> bool:
    s3 = boto3.client("s3", config=Config(signature_version=UNSIGNED))
    try:
        s3.download_file(bucket, key, str(dest))
        return True
    except ClientError:
        return False


@dataclass
class DataArtifact:
    """A data artifact that can be fetched from S3 or Zenodo.

    Downloads are attempted first from S3 (fast, no auth required), then from
    Zenodo as a fallback. The zip file is retained after extraction to indicate
    that the data was fetched rather than computed de novo. To force recomputation,
    delete the extracted files and set fetch=False.
    """

    filename: str
    extract_dir: Path
    fetch: bool = True

    @property
    def zip_path(self) -> Path:
        return self.extract_dir / self.filename

    @property
    def zenodo_url(self) -> str:
        return f"{ZENODO_BASE_URL}/{self.filename}"

    def download_if_missing(self) -> None:
        self.extract_dir.mkdir(parents=True, exist_ok=True)

        if not self.fetch:
            return

        if self.zip_path.exists():
            return

        if not _download_from_s3(PUBLIC_S3_BUCKET, self.filename, self.zip_path):
            console.print("[yellow]S3 download failed, falling back to Zenodo[/yellow]")
            subprocess.run(
                ["wget", "-O", str(self.zip_path), self.zenodo_url],
                check=True,
            )

        with zipfile.ZipFile(self.zip_path, "r") as zf:
            zf.extractall(self.extract_dir)
