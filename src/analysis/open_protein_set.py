from collections.abc import Sequence
from functools import partial
from pathlib import Path

import boto3
from botocore import UNSIGNED
from botocore.client import BaseClient
from botocore.config import Config

from analysis.utils import progress


def get_s3_client() -> BaseClient:
    """Create an S3 client configured for unsigned requests."""
    return boto3.client("s3", config=Config(signature_version=UNSIGNED))


def fetch_pdb(id: str, db_dir: Path, s3_client: BaseClient) -> Path:
    """Downloads (if not found) and then returns local pdb path.

    If the file is not found in the provided database dir, it will be downloaded from
    https://registry.opendata.aws/openfold/.
    """
    pdb_path = db_dir / f"{id}.pdb"
    if pdb_path.exists():
        return pdb_path

    pdb_path.parent.mkdir(parents=True, exist_ok=True)

    s3_key = f"uniclust30/{id}/pdb/{id}.pdb"

    s3_client.download_file("openfold", s3_key, str(pdb_path))

    return pdb_path


def fetch_msa(id: str, db_dir: Path, s3_client: BaseClient) -> Path:
    """Downloads (if not found) and then returns local a3m path.

    If the file is not found in the provided database dir, it will be downloaded from
    https://registry.opendata.aws/openfold/.
    """
    a3m_path = db_dir / f"{id}.a3m"
    if a3m_path.exists():
        return a3m_path

    a3m_path.parent.mkdir(parents=True, exist_ok=True)

    s3_key = f"uniclust30/{id}/a3m/uniclust30.a3m"

    s3_client.download_file("openfold", s3_key, str(a3m_path))

    return a3m_path


def fetch_msas(ids: Sequence[str], db_dir: Path) -> dict[str, Path]:
    fetch = partial(fetch_msa, db_dir=db_dir, s3_client=get_s3_client())
    return {id: fetch(id) for id in progress(ids, desc="Downloading or fetching MSAs")}


def fetch_pdbs(ids: Sequence[str], db_dir: Path) -> dict[str, Path]:
    fetch = partial(fetch_pdb, db_dir=db_dir, s3_client=get_s3_client())
    return {id: fetch(id) for id in progress(ids, desc="Downloading or fetching PDBs")}


def fetch_all_ids(cache_file: Path) -> list[str]:
    if cache_file.exists():
        return cache_file.read_text().strip().split("\n")

    s3_client = get_s3_client()
    paginator = s3_client.get_paginator("list_objects_v2")
    pages = paginator.paginate(Bucket="openfold", Prefix="uniclust30/")

    ids = []
    for page in pages:
        for obj in page.get("Contents", []):
            if obj["Key"].endswith("/a3m/uniclust30.a3m"):
                id = obj["Key"].removeprefix("uniclust30/").removesuffix("/a3m/uniclust30.a3m")
                ids.append(id)

    cache_file.write_text("\n".join(ids) + "\n")

    return ids

if __name__ == "__main__":

    pdb_path = Path("./data/uniclust30/pdbs/")
    path = fetch_pdb("A0A0F9VGA8", pdb_path, get_s3_client())
    print(path)
    print(path.exists())
