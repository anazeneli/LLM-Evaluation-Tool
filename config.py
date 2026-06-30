"""Parses eval_config.yaml into typed structures used by submit_jobs.py."""

from __future__ import annotations

from dataclasses import dataclass, field

import yaml


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
    with open(path) as f:
        raw = yaml.safe_load(f)

    models = [
        ModelConfig(
            name=m["name"],
            path=m["path"],
            arch=m.get("arch", m["name"]),
        )
        for m in raw.get("models", [])
    ]
    prompts = [
        PromptConfig(name=p["name"], system=p.get("system", ""))
        for p in raw.get("prompts", [{"name": "default"}])
    ]
    return EvalConfig(
        experiment_name=raw["experiment"]["name"],
        dataset=raw["experiment"]["dataset"],
        machine=raw["job"]["machine"].upper().replace("-", "_"),
        interruptible=raw["job"].get("interruptible", False),
        max_model_len=raw["job"].get("max_model_len", 2048),
        models=models,
        prompts=prompts,
    )
