# LLM Eval Template

Offline batch evaluation of fine-tuned models using Lightning Batch Jobs and vLLM.
Built for comparing model checkpoints across large prompt sets.

**The goal of this template is one model evaluation per LitLogger log.** Each
(model × prompt) combination runs as its own job and writes one self-contained
entry to a shared LitLogger experiment — its metrics, config, and full results
file. You compare runs side by side in the LitLogger dashboard.

A **sample dataset of 20 prompts** is included so you can run the full pipeline
immediately. Swap in your real dataset when ready.

---

## How it works

```
eval_config.yaml
      │
      └─ submit_jobs.py  ──snapshot──▶  Batch Job  ×  (models × prompts)
                                              │
                                              ├─ serve_model.py      LitServe + vLLM
                                              ├─ run_inference.py    concurrent inference
                                              └─ LitLogger           one log per model eval
```

**Each job runs on its own GPU, in parallel.** Lightning snapshots this Studio's
code and environment and deploys it to a fresh machine per job — no cluster to manage.
One job evaluates one model with one prompt over the **entire** dataset through a
single vLLM endpoint, and logs that run to LitLogger.

---

## Quickstart

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Point to your model checkpoints
Edit `eval_config.yaml` and set `models[*].path` to your checkpoint locations.
Paths under `/teamspace/` are shared across Studios and Jobs — no copy needed.

To download weights from HuggingFace:
```bash
python download_model.py --model-id google/gemma-3-4b-it --save-dir /teamspace/studios/this_studio/models/gemma_4b
```

### 3. Preview then submit
```bash
python submit_jobs.py --dry-run        # see what jobs would be submitted
python submit_jobs.py --limit 5        # smoke test: 5 prompts per job
python submit_jobs.py                  # full run
```

---

## Scaling to your real dataset

The sample dataset (`data/sample_prompts.jsonl`) has 20 prompts to verify
the pipeline. For production, point at your full dataset:

```yaml
# eval_config.yaml
experiment:
  dataset: /path/to/your/prompts.jsonl   # same JSONL format
```

**Throughput scales by using a more capable single GPU — not by splitting the
dataset.** Each job sends the whole dataset to one vLLM endpoint, and vLLM's
continuous batching already saturates that GPU on its own: it packs many prompts
into each forward pass and streams new ones in as others finish. The way to clear
the dataset faster is to give that one endpoint a bigger GPU. More memory bandwidth
and VRAM let vLLM hold a larger concurrent batch and decode it faster.

Reach for a stronger single GPU to go faster — e.g. `L4` → `H100`:

```yaml
# eval_config.yaml
job:
  machine: H100   # high-throughput single GPU for large datasets
```

> **Benchmark before assuming bigger is better.** A 4B model may underutilize an
> H100 — run a slice with `--limit` on each machine and compare `throughput_req_s`
> in LitLogger before committing a full sweep to expensive hardware.

> Multi-GPU machines exist, but this template does not split the dataset across
> GPUs. The only single-job multi-GPU option vLLM offers without splitting is
> tensor parallelism (one model spread across N GPUs behind one endpoint); a 4B
> model doesn't need it, so it isn't configured here.

---

## Dataset format

```jsonl
{"id": "001", "prompt": "Your prompt text here"}
{"id": "002", "prompt": "Another prompt"}
```

Or with pre-built messages (system prompt already included):
```jsonl
{"id": "001", "messages": [{"role": "system", "content": "..."}, {"role": "user", "content": "..."}]}
```

---

## Output

Each job writes a JSONL results file and logs a summary to LitLogger:

```jsonl
{"id": "001", "prompt": "...", "response": "...", "latency_s": 0.42, "error": null}
```

LitLogger tracks per-job: `num_prompts`, `throughput_req_s`, `latency_p50_s`, `latency_p95_s`, `error_rate`, and the full results file as an artifact. Compare all models in one experiment dashboard.

---

## Machine sizing

| Model size | Default machine | High-throughput option |
|---|---|---|
| ≤1B | L4 | H100 |
| 4B | L4 (default) | H100 |
| 12B | A100 | H100 |
| 27B+ | A100_80GB | H100 |

`L4` is the default — comfortable for a 4B model. `H100` is the high-throughput
single-GPU upgrade: more memory bandwidth and VRAM let vLLM run a larger concurrent
batch and clear the dataset faster on one endpoint. For a small model on a large
dataset, sizing up to an H100 is the lever to pull.

> **Note:** Gemma3 models require L4 or better — T4 lacks the FlashAttention memory bandwidth needed by this vLLM version.

Set `job.machine` in `eval_config.yaml`. Use `--limit 5` for a quick sanity check before committing to a full sweep on expensive hardware.

---

## Files

| File | Purpose |
|---|---|
| `eval_config.yaml` | **Start here** — model paths, prompts, machine |
| `data/sample_prompts.jsonl` | 20-prompt sample dataset — replace with your data |
| `submit_jobs.py` | Submit one batch job per (model × prompt) |
| `run_job.py` | Job entry point: start server → run inference → log |
| `run_inference.py` | Streams prompts, concurrent requests, writes results |
| `serve_model.py` | LitServe + vLLM inference server |
| `download_model.py` | Download weights from HuggingFace |
| `requirements.txt` | Python dependencies |
