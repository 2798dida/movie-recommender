from pathlib import Path
from tempfile import TemporaryDirectory
from urllib.request import urlretrieve
from zipfile import ZipFile

from src.config import ROOT_DIR


def download_and_extract_artifacts(zip_url: str) -> None:
    """Download artifact ZIP and extract it safely into the project root.

    The ZIP should contain paths such as:
    - models/als_model.joblib
    - models/user_item.npz
    - data/processed/movies.parquet
    """
    zip_url = zip_url.strip()
    if not zip_url:
        return

    root = ROOT_DIR.resolve()
    with TemporaryDirectory() as tmp_dir:
        zip_path = Path(tmp_dir) / "artifacts.zip"
        urlretrieve(zip_url, zip_path)

        with ZipFile(zip_path) as zip_file:
            for member in zip_file.infolist():
                target = (root / member.filename).resolve()
                if root != target and root not in target.parents:
                    raise ValueError(f"Path tidak aman di dalam ZIP: {member.filename}")
            zip_file.extractall(root)

