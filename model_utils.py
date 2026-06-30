"""Shared model path resolution used by serve_model.py and run_job.py."""

from __future__ import annotations

import os

# Models downloaded from litmodels land here — on teamspace shared storage so they
# persist across job runs and don't need to be re-downloaded each time.
MODELS_DIR = os.path.join(os.path.dirname(__file__), "models")


def resolve_model_path(path: str) -> str:
    """Return a local filesystem path, downloading from litmodels if needed.

    Resolution order:
      1. Absolute path or existing local directory → use as-is.
      2. Three-part ref with no leading dots (org/teamspace/model) → litmodels download to MODELS_DIR.
      3. Anything else (HuggingFace ID, relative path) → pass through to vLLM.
    """
    if os.path.isabs(path) or os.path.isdir(path):
        return path

    parts = path.split("/")
    if len(parts) != 3 or any(p.startswith(".") for p in parts):
        return path

    import litmodels

    model_name = parts[-1]
    local_dir = os.path.join(MODELS_DIR, model_name)
    if os.path.isfile(os.path.join(local_dir, "config.json")):
        print(f"[model] Using cached model at: {local_dir}")
    else:
        print(f"[model] Downloading from litmodels: {path} → {local_dir}")
        os.makedirs(local_dir, exist_ok=True)
        litmodels.download_model(path, download_dir=local_dir)
        print(f"[model] Model ready at: {local_dir}")
    return local_dir
