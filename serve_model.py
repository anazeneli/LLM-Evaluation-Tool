"""Local LitServe model server backed by vLLM.

Loads a model from disk (or a Hugging Face model ID) and exposes it on
POST /predict with server-side micro-batching. vLLM handles continuous
batching and paged attention internally, so concurrent eval requests are
grouped and processed together without any Ray dependency.

Usage:
    python serve_model.py
    python serve_model.py --model-dir models/gemma --port 8080 --max-batch-size 32

Request body  (JSON):
    {"messages": [{"role": "system", "content": "..."}, {"role": "user", "content": "..."}],
     "temperature": 0.0, "max_new_tokens": 220}

Response body (JSON):
    {"output": {"text": "..."}}
"""

from __future__ import annotations

import argparse
import os

import litserve as ls
from vllm import LLM, SamplingParams


def _resolve_model_path(path: str) -> str:
    """Return a local filesystem path, pulling from the litmodels registry if needed.

    A registry reference is anything that is not an absolute path and does not exist
    as a local file or directory (e.g. 'org/teamspace/gemma-3-1b-it').
    Absolute paths (/teamspace/...) and existing relative paths pass through unchanged.
    """
    if os.path.isabs(path) or os.path.exists(path):
        return path
    import litmodels

    print(f"[serve_model] Resolving litmodels reference: {path}")
    local = litmodels.download_model(path, download_dir="/tmp/models")
    print(f"[serve_model] Model ready at: {local}")
    return str(local)


class VLLMLitAPI(ls.LitAPI):
    def __init__(
        self,
        model_dir: str,
        max_model_len: int = 2048,
        max_batch_size: int = 32,
        batch_timeout: float = 0.05,
        gpu_memory_utilization: float = 0.90,
    ) -> None:
        super().__init__(max_batch_size=max_batch_size, batch_timeout=batch_timeout)
        self.model_dir = model_dir
        self.max_model_len = max_model_len
        self.gpu_memory_utilization = gpu_memory_utilization

    def setup(self, device: str) -> None:
        self.llm = LLM(
            model=self.model_dir,
            max_model_len=self.max_model_len,
            dtype="auto",
            enforce_eager=False,  # T4 cannot run Gemma3 inference — use L4 or better (see README).
            # Set to True only if hitting CUDA graph warmup errors on other hardware.
            gpu_memory_utilization=self.gpu_memory_utilization,
        )
        print(f"vLLM engine ready: {self.model_dir}")

    def decode_request(self, request: dict) -> dict:
        return request

    def predict(self, requests: list[dict]) -> list[str]:
        conversations = [r.get("messages", []) for r in requests]
        # All eval requests share the same sampling params; use first request's values.
        max_tokens = int(requests[0].get("max_new_tokens", 220))
        temperature = float(requests[0].get("temperature", 0.0))

        sampling_params = SamplingParams(
            temperature=temperature,
            max_tokens=max_tokens,
        )
        outputs = self.llm.chat(conversations, sampling_params=sampling_params)
        return [o.outputs[0].text.strip() for o in outputs]

    def encode_response(self, outputs: list[str]) -> list[dict]:
        return [{"text": o} for o in outputs]


def main() -> None:
    parser = argparse.ArgumentParser(description="Serve a local model with LitServe + vLLM")
    parser.add_argument(
        "--model-dir",
        default=os.environ.get("GEMMA_MODEL_DIR", "models/gemma"),
        help="Local path or Hugging Face model ID",
    )
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument(
        "--max-batch-size",
        type=int,
        default=int(os.environ.get("SERVER_MAX_BATCH_SIZE", "32")),
        help="Max requests to batch before forwarding to vLLM",
    )
    parser.add_argument(
        "--batch-timeout",
        type=float,
        default=float(os.environ.get("SERVER_BATCH_TIMEOUT", "0.05")),
        help="Seconds to wait collecting a batch before flushing",
    )
    parser.add_argument(
        "--max-model-len",
        type=int,
        default=int(os.environ.get("VLLM_MAX_MODEL_LEN", "2048")),
    )
    parser.add_argument(
        "--gpu-memory-utilization",
        type=float,
        default=float(os.environ.get("VLLM_GPU_MEMORY_UTILIZATION", "0.90")),
        help="Fraction of GPU memory vLLM may use (lower when sharing GPU between two servers)",
    )
    args = parser.parse_args()
    model_dir = _resolve_model_path(args.model_dir)

    api = VLLMLitAPI(
        model_dir=model_dir,
        max_model_len=args.max_model_len,
        max_batch_size=args.max_batch_size,
        batch_timeout=args.batch_timeout,
        gpu_memory_utilization=args.gpu_memory_utilization,
    )
    server = ls.LitServer(api, accelerator="auto")
    server.run(port=args.port)


if __name__ == "__main__":
    main()
