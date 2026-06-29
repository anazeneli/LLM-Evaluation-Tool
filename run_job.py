"""Batch job entry point — runs inside a Lightning Batch Job.

Starts the vLLM server, runs inference over the full prompt dataset,
logs results to LitLogger, then exits cleanly.

Usage (local smoke test):
    python run_job.py \\
        --model-dir models/gemma_ft_v1 \\
        --model-name ft_v1 \\
        --dataset prompts.jsonl \\
        --system-prompt "You are a helpful assistant." \\
        --limit 10

Launched automatically by submit_jobs.py for each (model, prompt) pair.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time

import requests


def _wait_for_server(url: str, timeout: int = 300) -> bool:
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            r = requests.get(f"{url}/health", timeout=2)
            if r.status_code == 200:
                return True
        except Exception:
            pass
        time.sleep(3)
    return False


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--model-name", required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--system-prompt", default="")
    parser.add_argument("--experiment-name", default=os.environ.get("EXPERIMENT_NAME", "batch-eval"))
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument("--max-model-len", type=int, default=int(os.environ.get("VLLM_MAX_MODEL_LEN", "2048")))
    parser.add_argument("--max-workers", type=int, default=64)
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()

    base_url = f"http://localhost:{args.port}"
    output_path = f"results/{args.model_name}.jsonl"

    print(f"[run_job] Starting server: {args.model_dir}")
    server = subprocess.Popen(
        [
            sys.executable, "serve_model.py",
            "--model-dir", args.model_dir,
            "--port", str(args.port),
            "--max-model-len", str(args.max_model_len),
        ],
        start_new_session=True,  # own process group so killpg reaches vLLM child processes
    )

    print("[run_job] Waiting for server...")
    if not _wait_for_server(base_url):
        server.terminate()
        raise SystemExit("[run_job] Server failed to start within 5 minutes.")

    print("[run_job] Server ready. Running inference...")
    cmd = [
        sys.executable, "run_inference.py",
        "--model-url", base_url,
        "--model-name", args.model_name,
        "--dataset", args.dataset,
        "--output", output_path,
        "--experiment-name", args.experiment_name,
        "--max-workers", str(args.max_workers),
        "--log",
    ]
    if args.system_prompt:
        cmd += ["--system-prompt", args.system_prompt]
    if args.limit > 0:
        cmd += ["--limit", str(args.limit)]

    try:
        result = subprocess.run(cmd, check=True)
    finally:
        print("[run_job] Shutting down server...")
        os.killpg(os.getpgid(server.pid), 15)  # SIGTERM to entire process group (catches vLLM child)
        try:
            server.wait(timeout=30)
        except subprocess.TimeoutExpired:
            os.killpg(os.getpgid(server.pid), 9)  # SIGKILL if it didn't exit cleanly

    sys.exit(result.returncode)


if __name__ == "__main__":
    main()
