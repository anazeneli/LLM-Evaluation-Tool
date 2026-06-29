"""Run a large prompt dataset through a LitServe/vLLM endpoint.

Streams the JSONL row by row (100k+ scale), fans out concurrent requests,
writes results to a JSONL file, and logs throughput + latency summary to
LitLogger.

Dataset format (one JSON object per line):
    {"id": "001", "prompt": "Classify this support ticket: ..."}

Or with pre-built messages (system prompt already included):
    {"id": "001", "messages": [{"role": "user", "content": "..."}]}

Usage:
    python run_inference.py \\
        --model-url http://localhost:8080 \\
        --dataset prompts.jsonl \\
        --model-name gemma_ft_v1 \\
        --system-prompt "You are a helpful assistant."
"""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import os
import statistics
import time
from pathlib import Path

import requests


def _call(url: str, row: dict, system_prompt: str, prompt_field: str, timeout: int) -> dict:
    start = time.perf_counter()
    if "messages" in row:
        messages = row["messages"]
    else:
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": row[prompt_field]})

    try:
        r = requests.post(
            f"{url}/predict",
            json={"messages": messages, "temperature": 0.0, "max_new_tokens": 512},
            timeout=timeout,
        )
        r.raise_for_status()
        data = r.json()
        if isinstance(data, dict):
            out = data.get("output", data)
            response = out.get("text", str(out)) if isinstance(out, dict) else str(out)
        elif isinstance(data, list):
            response = data[0].get("text", str(data[0])) if data else ""
        else:
            response = str(data)
        error = None
    except Exception as e:
        response = ""
        error = f"{type(e).__name__}: {e} | body={getattr(r, 'text', 'N/A')[:200]}"

    return {
        "id": row.get("id", ""),
        "prompt": row.get(prompt_field, ""),
        "response": response,
        "latency_s": round(time.perf_counter() - start, 3),
        "error": error,
    }


def run(
    model_url: str,
    dataset_path: str,
    output_path: str,
    model_name: str,
    system_prompt: str,
    prompt_field: str,
    max_workers: int,
    timeout: int,
    limit: int | None,
    experiment_name: str,
    log: bool,
) -> None:
    rows = []
    with open(dataset_path) as f:
        for i, line in enumerate(f):
            if limit and i >= limit:
                break
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))

    print(f"[run_inference] {len(rows)} prompts | model: {model_name} | workers: {max_workers}")

    results = []
    errors = 0
    latencies = []
    t0 = time.perf_counter()

    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = {
            pool.submit(_call, model_url, row, system_prompt, prompt_field, timeout): row
            for row in rows
        }
        for i, future in enumerate(concurrent.futures.as_completed(futures), 1):
            result = future.result()
            results.append(result)
            latencies.append(result["latency_s"])
            if result["error"]:
                errors += 1
                if errors <= 3:
                    print(f"  [error] id={result['id']} | {result['error']}")
            if i % 500 == 0 or i == len(rows):
                elapsed = time.perf_counter() - t0
                print(f"  {i}/{len(rows)} | {elapsed:.1f}s elapsed | {i/elapsed:.1f} req/s | errors: {errors}")

    elapsed_total = time.perf_counter() - t0
    throughput = len(rows) / elapsed_total

    summary = {
        "model": model_name,
        "num_prompts": len(rows),
        "errors": errors,
        "error_rate": round(errors / len(rows), 4),
        "throughput_req_s": round(throughput, 2),
        "latency_p50_s": round(statistics.median(latencies), 3),
        "latency_p95_s": round(sorted(latencies)[int(len(latencies) * 0.95)], 3),
        "elapsed_s": round(elapsed_total, 1),
    }
    print(f"\n[run_inference] Done. {summary}")

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        for r in results:
            f.write(json.dumps(r) + "\n")
    print(f"[run_inference] Results → {output_path}")

    if log:
        try:
            from litlogger import Experiment
            exp = Experiment(name=experiment_name)
            metrics = {k: float(v) for k, v in summary.items() if isinstance(v, (int, float))}
            exp.log_metrics(metrics)
            exp.log_metadata({"model": model_name})
            exp.log_file(output_path)
            exp.print_url()
        except Exception as e:
            print(f"[run_inference] LitLogger warning: {e}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Batch inference runner for LitServe + vLLM")
    parser.add_argument("--model-url", default="http://localhost:8080")
    parser.add_argument("--model-name", default="model")
    parser.add_argument("--dataset", required=True, help="JSONL file of prompts")
    parser.add_argument("--output", default="results/output.jsonl", help="Where to write responses")
    parser.add_argument("--system-prompt", default="", help="System prompt prepended to every row")
    parser.add_argument("--prompt-field", default="prompt", help="JSONL field containing the user prompt")
    parser.add_argument("--max-workers", type=int, default=int(os.environ.get("MAX_WORKERS", "64")))
    parser.add_argument("--timeout", type=int, default=60)
    parser.add_argument("--limit", type=int, default=None, help="Cap rows (for smoke tests)")
    parser.add_argument("--log", action="store_true", help="Log summary metrics to LitLogger")
    parser.add_argument("--experiment-name", default=os.environ.get("EXPERIMENT_NAME", "batch-eval"))
    args = parser.parse_args()

    run(
        model_url=args.model_url,
        dataset_path=args.dataset,
        output_path=args.output,
        model_name=args.model_name,
        system_prompt=args.system_prompt,
        prompt_field=args.prompt_field,
        max_workers=args.max_workers,
        timeout=args.timeout,
        limit=args.limit,
        experiment_name=args.experiment_name,
        log=args.log,
    )


if __name__ == "__main__":
    main()
