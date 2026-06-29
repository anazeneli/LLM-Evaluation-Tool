"""Submit one Lightning Batch Job per (model, prompt) combination.

Each job loads one checkpoint with one prompt variant, runs the full benchmark,
and logs to a shared LitLogger experiment. Compare all combinations in the
LitLogger dashboard after jobs complete.

Usage:
    python submit_jobs.py                          # uses eval_config.yaml
    python submit_jobs.py --config my_config.yaml
    python submit_jobs.py --config eval_config.yaml --limit 3  # smoke test
"""

from __future__ import annotations

import argparse
import time

import yaml
from lightning_sdk import Studio
from lightning_sdk.studio import Machine


def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="eval_config.yaml")
    parser.add_argument("--limit", type=int, default=0, help="Row cap per job (0 = all); for smoke tests")
    parser.add_argument("--dry-run", action="store_true", help="Print jobs without submitting")
    args = parser.parse_args()

    cfg = load_config(args.config)

    experiment_name = cfg["experiment"]["name"]
    dataset = cfg["experiment"]["dataset"]
    machine_str = cfg["job"]["machine"].upper().replace("-", "_")
    machine = getattr(Machine, machine_str, Machine.L4)
    interruptible = cfg["job"].get("interruptible", False)
    max_model_len = cfg["job"].get("max_model_len", 2048)

    models = cfg.get("models", [])
    prompts = cfg.get("prompts", [{"name": "default", "system": ""}])

    studio = Studio()
    total_jobs = len(models) * len(prompts)
    print(f"Studio    : {studio.name}")
    print(f"Experiment: {experiment_name}")
    print(f"Machine   : {machine_str} → {machine}")
    print(f"Models    : {len(models)}")
    print(f"Prompts   : {len(prompts)}")
    print(f"Total jobs: {total_jobs}\n")

    submitted = []
    for model in models:
        for prompt in prompts:
            model_name = f"{model['name']}__{prompt['name']}"
            job_name = f"eval-{model_name}-{int(time.time())}"

            cmd_parts = [
                "python run_job.py",
                f"--model-dir {model['path']}",
                f"--model-name {model_name}",
                f"--experiment-name {experiment_name}",
                f"--dataset {dataset}",
                f"--max-model-len {max_model_len}",
            ]
            if prompt.get("system"):
                prompt_arg = prompt["system"].strip().replace("\n", " ")
                cmd_parts.append(f"--system-prompt \"{prompt_arg}\"")
            if args.limit > 0:
                cmd_parts.append(f"--limit {args.limit}")

            command = " ".join(cmd_parts)

            print(f"  Job: {job_name}")
            print(f"       model  = {model['path']}")
            print(f"       prompt = {prompt['name']}")
            print(f"       cmd    = {command}")

            if args.dry_run:
                print("       [dry-run — not submitted]\n")
                continue

            studio.run_job(
                name=job_name,
                machine=machine,
                command=command,
                interruptible=interruptible,
            )
            submitted.append(job_name)
            print(f"       submitted ✓\n")

    if not args.dry_run:
        print(f"{len(submitted)} job(s) submitted.")
        print(f"Track results in LitLogger under experiment: '{experiment_name}'")
    else:
        print(f"{total_jobs} job(s) would be submitted (dry run).")


if __name__ == "__main__":
    main()
