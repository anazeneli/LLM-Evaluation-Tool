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

### 2. Upload model weights to the Lightning model registry (once per teamspace)
Run this once from your Studio. Batch Jobs can't authenticate interactively, so weights must be
uploaded ahead of time. The Lightning model registry is shared across every Studio and Batch Job
in your teamspace — one upload, no re-downloading per user or per job.

```bash
python download_model.py \
    --model-id unsloth/gemma-3-270m-it \
    --litmodels-name org/teamspace/gemma-3-270m-it
```

Replace `org/teamspace` with your Lightning organization and teamspace names.
If the model is already in the registry, the script skips the download automatically.
For gated models (e.g. `google/gemma-3-4b-it`) set `export HF_TOKEN="hf_..."` first.

### 3. Point to your model checkpoints
Edit `eval_config.yaml` and set `models[*].path` to the litmodels name you used above, and
`models[*].arch` for dashboard grouping.

```yaml
models:
  - name: baseline
    path: org/teamspace/gemma-3-270m-it        # litmodels registry name — downloaded at job start
    arch: gemma-3-270m-it
  - name: finetune_v1
    path: /teamspace/studios/this_studio/models/gemma-3-270m-it  # local path — used as-is
    arch: gemma-3-270m-it
```

Both path styles are supported. A registry name (`org/teamspace/...`) is pulled from litmodels at
job startup — shared across the whole teamspace, no re-downloading per user or per job. An absolute
path is passed straight to vLLM unchanged.

### 4. Preview then submit
```bash
python submit_jobs.py --dry-run        # see what jobs would be submitted
python submit_jobs.py --limit 5        # smoke test: spins up an L4, runs 5 prompts, shuts down
python submit_jobs.py                  # full run
```

> **Note:** vLLM requires a GPU and jobs cannot run on CPU. The minimum is an **L4** — T4 lacks
> the shared memory Gemma3's Triton attention kernels require (80KB needed, 64KB available).
> Use `--limit 5` for a cheap end-to-end smoke test before committing to a full sweep.

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

LitLogger tracks per-job:

- **Metrics:** `num_prompts`, `throughput_req_s`, `latency_p50_s`, `latency_p95_s`, `error_rate`
- **Tags:** `arch` (model family, sortable column for cross-run comparison), `experiment_type` (run identity: `{model}__{prompt}`)
- **Artifact:** full results JSONL file

**How experiments map to LitLogger:** `experiment.name` in `eval_config.yaml` becomes a LitLogger
experiment in your teamspace — think of it as a folder that groups runs for comparison. All jobs
submitted from the same config share one experiment. Each job logs as a separate run under
`{model}__{prompt}` (e.g. `google-base__default`, `google-it__default`) so runs never overwrite
each other. View all runs side-by-side at **lightning.ai → your teamspace → LitLogger**.

**The logging is yours to extend.** All LitLogger calls are in one place — the `if log:` block at the bottom of `run()` in `run_inference.py`. Add metrics to the `summary` dict, add tags to `log_metadata`, or call `exp.log_file()` with additional artifacts. Changes apply to every job without touching the submission or job-runner code.

---

## Machine sizing

| Model size | Default machine | High-throughput option |
|---|---|---|
| 270M | L4 | A100 |
| ≤1B | L4 | H100 |
| 4B | L4 | H100 |
| 12B | A100 | H100 |
| 27B+ | A100_80GB | H100 |

`L4` is the minimum — T4's shared memory is too small for Gemma3's attention kernels. `A100` or better is recommended for 1B+ models. `H100` is the high-throughput
single-GPU upgrade: more memory bandwidth and VRAM let vLLM run a larger concurrent
batch and clear the dataset faster on one endpoint.

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
