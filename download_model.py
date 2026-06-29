"""Download model weights from HuggingFace and optionally upload to the Lightning model registry.

Downloading to litmodels is the recommended path: the registry is shared across every Studio
and Batch Job in your teamspace, so jobs can pull the weights at runtime without any file
copying or credential setup.

Usage:
    # Download from HF and cache in the Lightning model registry (recommended):
    python download_model.py \
        --model-id google/gemma-3-4b-it \
        --litmodels-name org/teamspace/gemma-3-4b-it

    # Download to a local directory only (one-off / offline use):
    python download_model.py --model-id google/gemma-3-4b-it --save-dir /tmp/gemma-3-4b-it

For gated models (e.g. google/gemma-3-4b-it) set your HF token first:
    export HF_TOKEN="hf_..."
Public models (e.g. unsloth/gemma-3-4b-it) work without a token.
"""

from __future__ import annotations

import argparse
import os
import tempfile
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description="Download a model from HuggingFace")
    parser.add_argument(
        "--model-id",
        default=os.environ.get("GEMMA_MODEL_ID", "unsloth/gemma-3-1b-it"),
        help="HuggingFace model ID",
    )
    parser.add_argument(
        "--litmodels-name",
        default="",
        help="Upload to Lightning model registry after download. Format: org/teamspace/model-name",
    )
    parser.add_argument(
        "--save-dir",
        default="",
        help="Save to a local directory instead of (or in addition to) litmodels. "
             "Defaults to a temp directory when --litmodels-name is set.",
    )
    parser.add_argument(
        "--hf-token",
        default=os.environ.get("HF_TOKEN", ""),
    )
    args = parser.parse_args()

    # Resolve save directory
    _tmp_dir = None
    if args.save_dir:
        save_dir = Path(args.save_dir)
    elif args.litmodels_name:
        _tmp_dir = tempfile.mkdtemp(prefix="litmodels_download_")
        save_dir = Path(_tmp_dir)
    else:
        save_dir = Path(os.environ.get("GEMMA_MODEL_DIR", "models/gemma"))

    if (save_dir / "config.json").exists():
        print(f"Model already present at {save_dir}.")
    else:
        from huggingface_hub import snapshot_download

        token = args.hf_token or None
        if not token:
            print("No HF_TOKEN set — proceeding without authentication (public models only).")

        save_dir.mkdir(parents=True, exist_ok=True)
        print(f"Downloading {args.model_id} → {save_dir} ...")
        snapshot_download(repo_id=args.model_id, local_dir=str(save_dir), token=token)
        print(f"Download complete: {save_dir}")

    if args.litmodels_name:
        import litmodels

        print(f"Uploading {save_dir} → litmodels:{args.litmodels_name} ...")
        litmodels.upload_model(name=args.litmodels_name, model=str(save_dir))
        print(f"Model available as: {args.litmodels_name}")
        print("Use this name as the model path in eval_config.yaml.")


if __name__ == "__main__":
    main()
