from __future__ import annotations

import argparse
import os
import shutil
import tempfile
from pathlib import Path

from huggingface_hub import HfApi


def deploy_space(
    space_id: str,
    protein_repo: str,
    molecule_repo: str,
    affinity_repo: str,
    token: str,
    model_token: str = "",
    private: bool = False,
) -> str:
    root = Path(__file__).resolve().parents[1]
    space_source = root / "space"
    package_source = root / "src" / "affinity"
    if not (space_source / "Dockerfile").exists():
        raise FileNotFoundError(f"Missing Dockerfile under {space_source}")
    if not package_source.exists():
        raise FileNotFoundError(f"Missing affinity package under {package_source}")

    api = HfApi(token=token)
    for repo_id in (protein_repo, molecule_repo, affinity_repo):
        api.model_info(repo_id, token=token)

    api.create_repo(
        repo_id=space_id,
        repo_type="space",
        space_sdk="docker",
        private=private,
        exist_ok=True,
    )
    variables = {
        "PROTEIN_ONNX_REPO": protein_repo,
        "MOLECULE_ONNX_REPO": molecule_repo,
        "AFFINITY_MODEL_REPO": affinity_repo,
        "ONNX_DEVICE": "cpu",
    }
    for key, value in variables.items():
        api.add_space_variable(
            repo_id=space_id,
            key=key,
            value=value,
            description="Configured by the affinity deployment script",
        )
    if model_token:
        api.add_space_secret(
            repo_id=space_id,
            key="HF_TOKEN",
            value=model_token,
            description="Read access for private model repositories",
        )

    with tempfile.TemporaryDirectory(prefix="affinity-space-") as temporary:
        staging = Path(temporary)
        shutil.copytree(space_source, staging, dirs_exist_ok=True)
        shutil.copytree(package_source, staging / "affinity")
        api.upload_folder(
            repo_id=space_id,
            repo_type="space",
            folder_path=str(staging),
            commit_message="Deploy ESM-2, MoLFormer, and affinity ONNX application",
        )
    return f"https://huggingface.co/spaces/{space_id}"


def main() -> None:
    parser = argparse.ArgumentParser(description="Deploy the ONNX app to a Docker Space")
    parser.add_argument("--space", required=True, help="username/space-name")
    parser.add_argument("--protein-repo", required=True)
    parser.add_argument("--molecule-repo", required=True)
    parser.add_argument("--affinity-repo", required=True)
    parser.add_argument("--token", default=os.getenv("HF_TOKEN", ""))
    parser.add_argument(
        "--model-token",
        default=os.getenv("HF_MODEL_TOKEN", ""),
        help="Optional read token stored in the Space for private model repositories",
    )
    parser.add_argument("--private", action="store_true")
    args = parser.parse_args()
    if not args.token:
        parser.error("Provide --token or set HF_TOKEN")
    print(
        deploy_space(
            args.space,
            args.protein_repo,
            args.molecule_repo,
            args.affinity_repo,
            args.token,
            args.model_token,
            args.private,
        )
    )


if __name__ == "__main__":
    main()
