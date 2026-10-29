"""Upload a trained model folder to the Hugging Face Hub.

Run `hf auth login` first. No token is stored in this repository.
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

from huggingface_hub import HfApi, create_repo


ROOT = Path(__file__).parent
DEFAULT_MODEL_DIR = ROOT / "artifacts" / "hf" / "model"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Upload the Geek Genre Classifier.")
    parser.add_argument("--repo-id", required=True, help="Example: Ruaney/geek-genre-text-classifier")
    parser.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL_DIR)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.model_dir.exists():
        raise SystemExit("Model directory not found. Run train_hf.py first.")

    shutil.copyfile(ROOT / "MODEL_CARD.md", args.model_dir / "README.md")
    create_repo(args.repo_id, repo_type="model", exist_ok=True)
    HfApi().upload_folder(
        repo_id=args.repo_id,
        repo_type="model",
        folder_path=str(args.model_dir),
        commit_message="Upload Geek Genre Classifier",
    )
    print(f"Uploaded: https://huggingface.co/{args.repo_id}")


if __name__ == "__main__":
    main()
