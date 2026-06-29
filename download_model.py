"""Download model weights into this studio using huggingface_hub.

vLLM can load directly from a local directory or a HF model ID, but
pre-downloading avoids network latency at server startup.

Usage:
    python download_model.py
    python download_model.py --model-id google/gemma-3-4b-it --save-dir models/gemma

For gated models (e.g. google/gemma-3-4b-it) set your HF token first:
    export HF_TOKEN="hf_..."
Public models (e.g. unsloth/gemma-3-4b-it) work without a token.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description="Download a model to local storage")
    parser.add_argument(
        "--model-id",
        default=os.environ.get("GEMMA_MODEL_ID", "google/gemma-3-4b-it"),
    )
    parser.add_argument(
        "--save-dir",
        default=os.environ.get("GEMMA_MODEL_DIR", "models/gemma"),
    )
    parser.add_argument(
        "--hf-token",
        default=os.environ.get("HF_TOKEN", ""),
    )
    args = parser.parse_args()

    save_dir = Path(args.save_dir)

    if (save_dir / "config.json").exists():
        print(f"Model already present at {save_dir}. Delete the directory to re-download.")
        return

    from huggingface_hub import snapshot_download

    token = args.hf_token or None  # None lets huggingface_hub use cached login; works for public models
    if not token:
        print("No HF_TOKEN set — proceeding without authentication (public models only).")

    save_dir.mkdir(parents=True, exist_ok=True)
    print(f"Downloading {args.model_id} → {save_dir} ...")
    snapshot_download(repo_id=args.model_id, local_dir=str(save_dir), token=token)
    print(f"Done. Model saved to {save_dir}")


if __name__ == "__main__":
    main()
