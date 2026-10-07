"""Download SIFT1M and GIST1M into data/.

    python scripts/download_datasets.py --dataset all

SIFT1M is tried from a Hugging Face mirror first and falls back to the original TEXMEX FTP
archive. GIST1M comes from the TEXMEX FTP archive. A dataset already present is skipped. SIFT1M
takes about 0.5 GB on disk and GIST1M about 5.5 GB.
"""

from __future__ import annotations

import argparse
import shutil
import sys
import tarfile
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # make `src` importable

from src.config import DATA_DIR  # noqa: E402

SIFT1M_FTP = "ftp://ftp.irisa.fr/local/texmex/corpus/sift.tar.gz"
SIFT1M_HF = "https://huggingface.co/datasets/qbo-odp/sift1m/resolve/main"
GIST1M_FTP = "ftp://ftp.irisa.fr/local/texmex/corpus/gist.tar.gz"


def _download(url: str, dest: Path) -> None:
    """Download `url` to the file `dest`, creating its folder if needed."""
    print(f"Downloading {url} to {dest} ...")
    dest.parent.mkdir(parents=True, exist_ok=True)
    with urllib.request.urlopen(url) as resp, open(dest, "wb") as f:
        shutil.copyfileobj(resp, f)
    print(f"  done ({dest.stat().st_size / 1e6:.1f} MB)")


def download_sift1m(data_dir: Path) -> None:
    """Download SIFT1M into `data_dir`/sift, from the mirror or else from the FTP archive."""
    sift_dir = data_dir / "sift"
    if (sift_dir / "sift_base.fvecs").exists():
        print("SIFT1M already present; skipping.")
        return
    sift_dir.mkdir(parents=True, exist_ok=True)

    files = ["sift_base.fvecs", "sift_query.fvecs", "sift_groundtruth.ivecs", "sift_learn.fvecs"]
    try:
        for fname in files:
            _download(f"{SIFT1M_HF}/{fname}", sift_dir / fname)
        return
    except Exception as e:
        print(f"HuggingFace failed: {e}; trying FTP fallback...")

    archive = data_dir / "sift.tar.gz"
    _download(SIFT1M_FTP, archive)
    with tarfile.open(archive, "r:gz") as tar:
        tar.extractall(data_dir)
    archive.unlink()


def download_gist1m(data_dir: Path) -> None:
    """Download GIST1M into `data_dir`/gist from the FTP archive."""
    gist_dir = data_dir / "gist"
    if (gist_dir / "gist_base.fvecs").exists():
        print("GIST1M already present; skipping.")
        return
    archive = data_dir / "gist.tar.gz"
    _download(GIST1M_FTP, archive)
    with tarfile.open(archive, "r:gz") as tar:
        tar.extractall(data_dir)
    archive.unlink()


def main() -> None:
    """Parse the arguments and download the requested datasets."""
    parser = argparse.ArgumentParser(description="Download ANN benchmark datasets")
    parser.add_argument(
        "--dataset", choices=["sift1m", "gist1m", "all"], default="sift1m"
    )
    parser.add_argument("--data-dir", type=Path, default=DATA_DIR)
    args = parser.parse_args()

    if args.dataset in ("sift1m", "all"):
        download_sift1m(args.data_dir)
    if args.dataset in ("gist1m", "all"):
        download_gist1m(args.data_dir)


if __name__ == "__main__":
    main()
