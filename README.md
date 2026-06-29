# LLM Eval Template

Offline batch evaluation of fine-tuned models using Lightning Batch Jobs and vLLM.
Built for comparing model checkpoints across large prompt sets (100k+).

A **sample dataset of 20 prompts** is included so you can run the full pipeline
immediately. Swap in your real dataset when ready.

---

## How it works

```
eval_config.yaml
      │
      └─ submit_jobs.py  ──snapshot──▶  Batch Job  ×  (models × prompts × shards)
                                              │
                                              ├─ serve_model.py      LitServe + vLLM
                                              ├─ run_inference.py    concurrent inference
                                              └─ LitLogger           metrics + results artifact
```

**Each job runs on its own GPU, in parallel.** Lightning snapshots this Studio's
code and environment and deploys it to a fresh machine per job — no cluster to manage.

**Sharding is the key scaling lever.** Set `job.shards: 10` and a 100k-prompt dataset
splits across 10 L4s automatically, each handling 10k rows. Wall-clock time drops 10×
at the same total GPU cost.

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
the pipeline. For production, replace it with your full dataset and increase shards:

```yaml
# eval_config.yaml
experiment:
  dataset: /path/to/your/prompts.jsonl   # 100k+ rows, same JSONL format

job:
  shards: 10   # splits dataset across 10 parallel GPUs — 10× faster wall-clock
```

`run_inference.py` streams the file row by row and picks its shard via round-robin,
so memory stays flat and no pre-counting is needed. `shards: 1` (default) is a
single job; `shards: N` launches N jobs each processing every Nth row.

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

LitLogger tracks per-job: `throughput_req_s`, `latency_p50_s`, `latency_p95_s`, `error_rate`, and the full results file as an artifact. Compare all models in one experiment dashboard.

---

## Machine sizing

| Model size | Recommended machine |
|---|---|
| ≤1B | L4 |
| 4B | L4 |
| 12B | A100 |
| 27B+ | A100_80GB |

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
