# local-llm-bench

Benchmarks **Ollama** against **llama.cpp** serving *the same GGUF weights*, driven
entirely through the stock **OpenAI Python SDK**, across a sweep of **GPU offload
levels**.

The two engines share an inference core, so running them head-to-head only means
something if the weights are identical. They are: Ollama stores models as plain
GGUF blobs, and the harness resolves that path and hands the same file to
`llama-server`. The only variable left is the server — and the GPU knob.

## Why the GPU knob matters here

On Apple Silicon, Metal offload is automatic and unified memory means there is no
host-to-device copy, so "uses the GPU" is invisible by default. Making it
*measurable* requires controlling how many transformer layers are offloaded:

| Engine | Knob | How it is set |
|---|---|---|
| llama.cpp | `-ngl N` | command-line flag on `llama-server` |
| Ollama | `num_gpu N` | `PARAMETER` baked into a derived model |

Ollama's OpenAI-compatible endpoint has no field for `num_gpu`, so the harness
generates one derived model per offload level (`bench-ngl0`, `bench-ngl16`, …).
Derived models reuse the base blobs, so this costs no extra disk.

`-ngl 0` is CPU-only; `-ngl 33` puts all 32 blocks plus the output layer on the
GPU. The gap between those two columns is the GPU's contribution, in tokens per
second.

## Setup

```bash
brew install ollama llama.cpp
ollama serve &                       # or: brew services start ollama
ollama pull llama3.1:8b

python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
```

## Run

The `bench` command wraps the common operations (installed as a symlink in
`~/.local/bin`, which `~/.zshrc` adds to `PATH`):

```bash
bench                # start the dashboard if needed, then open it
bench status         # is it up?
bench stop           # shut it down
bench sweep --quick  # run a benchmark sweep (args pass to bench.run)
```

Or drive the modules directly:

```bash
.venv/bin/python -m bench.run --quick    # ngl 0/16/33, 1 repeat
.venv/bin/python -m bench.run            # full sweep, 3 repeats
.venv/bin/python -m bench.chart          # table + results/benchmark.png
.venv/bin/python -m bench.serve --open   # stdlib dashboard at http://127.0.0.1:8000
.venv/bin/streamlit run bench/app.py     # Streamlit dashboard at http://127.0.0.1:8501
```

Two dashboards, deliberately. `bench.serve` is stdlib-only and renders whatever
is in `latest.json` — nothing to install, nothing to restart. `bench/app.py` is
the Streamlit one: it adds a run picker over every historical sweep in
`results/`, plus prompt and engine filters.

`bench.serve` re-reads `results/latest.json` on every request, so re-running a
sweep and refreshing the page is enough — no restart. It is stdlib-only and adds
no dependency.

Useful flags: `--ngl 0 33` to sweep specific levels, `--only ollama` /
`--only llamacpp` to run one engine, `--model <tag>` for a different model,
`--repeats N`, `--max-tokens N`.

## What gets measured

Each request is streamed, which splits the work into two phases that respond to
GPU offload very differently:

- **TTFT** (time to first token) — prompt ingestion, or *prefill*. Compute-bound
  and highly parallel, so it benefits most from the GPU. The harness uses a
  ~3,000-token prompt to make this phase dominate.
- **Decode throughput** (tokens/sec after the first) — memory-bandwidth-bound.
  Improves with offload, but less dramatically than prefill.

Reporting a single tokens-per-second number would average these together and hide
the more interesting half, so they are charted separately.

Each configuration is warmed up once (the first call pays for cache allocation,
not inference) and then timed `--repeats` times; the summary reports medians.

## Layout

```
bench/backends.py   process control: llama-server lifecycle, Ollama variants, GGUF resolution
bench/client.py     the OpenAI SDK driver and the two-phase timer
bench/run.py        sweep orchestration -> results/*.json
bench/chart.py      results -> table + two-panel PNG
bench/serve.py      localhost dashboard (stdlib http.server)
bench/web/          the dashboard page: SVG charts, hover readout, table view
bench/app.py        Streamlit dashboard: run picker, filters, Altair charts
bin/bench           the `bench` command (start/status/stop/sweep)
bench/gpumon.py     optional powermetrics sampler (needs sudo; not in the main path)
```

Results land in `results/` as timestamped JSON plus `latest.json`. Raw server
logs for each offload level land there too, which is where to look when a
configuration fails to start.
