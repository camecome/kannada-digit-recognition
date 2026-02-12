import os
import shutil
import zipfile
from pathlib import Path

import fire
from kaggle.api.kaggle_api_extended import KaggleApi

KAGGLE_COMPETITION = "Kannada-MNIST"


# export KAGGLE_API_TOKEN=KGAT_2a887cd3fa19173a67ce13939eab1583
def authenticate_kaggle() -> KaggleApi:
    if "KAGGLE_API_TOKEN" not in os.environ:
        raise RuntimeError(
            "KAGGLE_API_TOKEN is not set. "
            "Get your token from https://www.kaggle.com/<username>/account and "
            "set it via: export KAGGLE_API_TOKEN=<your_token>"
        )
    api = KaggleApi()
    api.authenticate()
    return api


def download_dataset(target_dir: Path):
    print("Downloading dataset...")
    api = authenticate_kaggle()
    api.competition_download_files(KAGGLE_COMPETITION, path=str(target_dir))
    return target_dir / f"{KAGGLE_COMPETITION}.zip"


def extract_zip(zip_path: Path, extract_to: Path):
    if not zip_path.exists():
        raise FileNotFoundError(f"{zip_path} does not exist.")
    print(f"Extracting {zip_path}...")
    with zipfile.ZipFile(zip_path, "r") as zip_ref:
        zip_ref.extractall(extract_to)


def organize_files(target_dir: Path):
    print("Organizing files...")
    sample_submission = target_dir / "sample_submission.csv"
    original_test = target_dir / "test.csv"
    dig_test = target_dir / "Dig-MNIST.csv"
    new_test = target_dir / "test.csv"

    for file in [sample_submission, original_test]:
        if file.exists():
            file.unlink()

    if dig_test.exists():
        shutil.move(str(dig_test), str(new_test))


def download(target_dir: str = "data"):
    target_dir = Path(target_dir)
    target_dir.mkdir(parents=True, exist_ok=True)

    zip_path = download_dataset(target_dir)
    extract_zip(zip_path, target_dir)
    organize_files(target_dir)

    if zip_path.exists():
        zip_path.unlink()

    print(f"Dataset ready at {target_dir}")


if __name__ == "__main__":
    fire.Fire(download)
