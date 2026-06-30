"""Parses eval_config.yaml into typed structures used by submit_jobs.py."""

from __future__ import annotations

import os
from dataclasses import dataclass

import yaml

PLACEHOLDER_PATH = "org/teamspace/model-name"
VALID_MACHINES = {"T4", "L4", "A10G", "A100", "A100_80GB", "H100"}


@dataclass
class ModelConfig:
    name: str
    path: str
    arch: str = ""


@dataclass
class PromptConfig:
    name: str
    system: str = ""


@dataclass
class EvalConfig:
    experiment_name: str
    dataset: str
    machine: str
    interruptible: bool
    max_model_len: int
    models: list[ModelConfig]
    prompts: list[PromptConfig]


def load_config(path: str) -> EvalConfig:
    if not os.path.exists(path):
        raise SystemExit(f"Config file not found: {path}")

    with open(path) as f:
        raw = yaml.safe_load(f)

    errors = []

    raw_models = raw.get("models", [])
    if not raw_models:
        errors.append("models: no models defined — add at least one entry under 'models:'")

    models = []
    for m in raw_models:
        if not m.get("name"):
            errors.append("models: every model entry must have a 'name' field")
        if not m.get("path"):
            errors.append(f"models[{m.get('name', '?')}]: missing 'path'")
        elif m["path"] == PLACEHOLDER_PATH:
            errors.append(
                f"models[{m['name']}].path is still the placeholder '{PLACEHOLDER_PATH}'.\n"
                f"  Fix: python download_model.py --model-id unsloth/gemma-3-270m-it --litmodels-name <org>/<teamspace>/<name>\n"
                f"       then set models[{m['name']}].path to the litmodels name you used."
            )
        models.append(ModelConfig(
            name=m.get("name", ""),
            path=m.get("path", ""),
            arch=m.get("arch", m.get("name", "")),
        ))

    dataset = raw.get("experiment", {}).get("dataset", "")
    if not dataset:
        errors.append("experiment.dataset: missing — set it to your JSONL prompt file path")
    elif not os.path.exists(dataset):
        errors.append(f"experiment.dataset: file not found: '{dataset}'")

    machine_raw = raw.get("job", {}).get("machine", "L4")
    machine = machine_raw.upper().replace("-", "_")
    if machine not in VALID_MACHINES:
        errors.append(
            f"job.machine: '{machine_raw}' is not valid — choose from: {', '.join(sorted(VALID_MACHINES))}"
        )

    if errors:
        lines = "\n".join(f"  x {e}" for e in errors)
        raise SystemExit(f"\nConfig errors in {path}:\n{lines}\n")

    prompts = [
        PromptConfig(name=p["name"], system=p.get("system", ""))
        for p in raw.get("prompts", [{"name": "default"}])
    ]

    return EvalConfig(
        experiment_name=raw["experiment"]["name"],
        dataset=dataset,
        machine=machine,
        interruptible=raw["job"].get("interruptible", False),
        max_model_len=raw["job"].get("max_model_len", 2048),
        models=models,
        prompts=prompts,
    )
