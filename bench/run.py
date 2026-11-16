"""Sweep both backends across GPU offload levels and write the raw samples.

Usage:
    python -m bench.run                 # full sweep
    python -m bench.run --quick         # one repeat, three offload levels
    python -m bench.run --ngl 0 33      # just CPU-only vs fully offloaded
"""
from __future__ import annotations

import argparse
import json
import platform
import statistics
import subprocess
import uuid
import time
from pathlib import Path

from .backends import (
    LlamaServer,
    OLLAMA_URL,
    gguf_path_for,
    ollama_unload,
    ollama_variant,
    wait_http,
)
from .client import Sample, client_for, measure

RESULTS = Path(__file__).resolve().parent.parent / "results"

# llama3.1:8b has 32 transformer blocks; 33 offloads the output layer too.
DEFAULT_NGL = [0, 8, 16, 24, 33]

SHORT_PROMPT = "Explain what a GPU kernel is, in one paragraph."

# A long prompt makes the prefill phase dominate, which is where GPU offload
# shows up most dramatically. Padding is filler by design -- we are timing
# token ingestion, not asking for a good answer.
_FILLER = (
    "The quick brown fox jumps over the lazy dog near the riverbank while "
    "the morning fog settles across the valley and the city slowly wakes. "
)
LONG_PROMPT = (
    "Read the following notes and then answer the question at the end.\n\n"
    + _FILLER * 110
    + "\n\nQuestion: in one paragraph, what is a GPU kernel?"
)

PROMPTS = {"short": SHORT_PROMPT, "long": LONG_PROMPT}


def _unique(prompt: str) -> str:
    """Prepend a nonce so the prompt has never been seen before.

    Both engines cache prompt KV by longest common prefix, so re-sending an
    identical prompt skips prefill entirely and TTFT collapses to cache-hit
    latency -- on a 1,500-token prompt that showed up as 0.07s on a CPU-only
    run, which is physically impossible. The nonce has to *lead* the prompt; a
    trailing one still matches the cached prefix and changes nothing.
    """
    return f"[session {uuid.uuid4().hex[:12]}]\n{prompt}"


def host_info() -> dict:
    def sysctl(key: str) -> str:
        r = subprocess.run(["sysctl", "-n", key], capture_output=True, text=True)
        return r.stdout.strip()

    try:
        ram_gb = int(sysctl("hw.memsize")) // 1024**3
    except ValueError:
        ram_gb = 0
    return {
        "chip": sysctl("machdep.cpu.brand_string") or platform.processor(),
        "ram_gb": ram_gb,
        "platform": platform.platform(),
        "python": platform.python_version(),
    }


def sweep_llamacpp(gguf: Path, model: str, ngl_values: list[int], repeats: int,
                   max_tokens: int) -> list[Sample]:
    samples: list[Sample] = []
    for ngl in ngl_values:
        print(f"  llama.cpp  -ngl {ngl:<3} starting server...", flush=True)
        with LlamaServer(gguf, ngl, log_dir=RESULTS) as base_url:
            client = client_for(base_url)
            # Warm-up: the first call after a load pays for buffer allocation,
            # not inference. A short prompt is enough -- warming with the long
            # one would buy an extra full prefill per level and measure nothing.
            measure(client, model, _unique(SHORT_PROMPT), backend="llama.cpp", ngl=ngl,
                    prompt_name="warmup", max_tokens=16)
            for name, prompt in PROMPTS.items():
                for _ in range(repeats):
                    s = measure(client, model, _unique(prompt), backend="llama.cpp", ngl=ngl,
                                prompt_name=name, max_tokens=max_tokens)
                    samples.append(s)
                    print(f"    {name:<5} prompt={s.prompt_tokens:>5}tok  ttft={s.ttft_s:7.3f}s  "
                          f"prefill={s.prefill_tps:8.1f} tok/s  decode={s.decode_tps:7.2f} tok/s",
                          flush=True)
    return samples


def sweep_ollama(model: str, ngl_values: list[int], repeats: int,
                 max_tokens: int) -> list[Sample]:
    samples: list[Sample] = []
    client = client_for(f"{OLLAMA_URL}/v1")
    for ngl in ngl_values:
        tag = ollama_variant(model, ngl)
        print(f"  ollama     num_gpu {ngl:<3} ({tag})", flush=True)
        try:
            measure(client, tag, _unique(SHORT_PROMPT), backend="ollama", ngl=ngl,
                    prompt_name="warmup", max_tokens=16)
            for name, prompt in PROMPTS.items():
                for _ in range(repeats):
                    s = measure(client, tag, _unique(prompt), backend="ollama", ngl=ngl,
                                prompt_name=name, max_tokens=max_tokens)
                    samples.append(s)
                    print(f"    {name:<5} prompt={s.prompt_tokens:>5}tok  ttft={s.ttft_s:7.3f}s  "
                          f"prefill={s.prefill_tps:8.1f} tok/s  decode={s.decode_tps:7.2f} tok/s",
                          flush=True)
        finally:
            ollama_unload(tag)
    return samples


def summarize(samples: list[Sample]) -> list[dict]:
    """Collapse repeats into medians -- robust to a single slow outlier."""
    buckets: dict[tuple, list[Sample]] = {}
    for s in samples:
        buckets.setdefault((s.backend, s.ngl, s.prompt_name), []).append(s)
    rows = []
    for (backend, ngl, prompt_name), group in sorted(buckets.items()):
        rows.append({
            "backend": backend,
            "ngl": ngl,
            "prompt": prompt_name,
            "n": len(group),
            "ttft_s": round(statistics.median(g.ttft_s for g in group), 4),
            "prefill_tps": round(statistics.median(g.prefill_tps for g in group), 1),
            "decode_tps": round(statistics.median(g.decode_tps for g in group), 2),
            "prompt_tokens": int(statistics.median(g.prompt_tokens for g in group)),
            "completion_tokens": int(statistics.median(g.completion_tokens for g in group)),
        })
    return rows


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="llama3.1:8b")
    ap.add_argument("--ngl", type=int, nargs="+", default=None,
                    help="GPU layer counts to sweep (default: 0 8 16 24 33)")
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--max-tokens", type=int, default=256)
    ap.add_argument("--quick", action="store_true", help="1 repeat, ngl 0/16/33")
    ap.add_argument("--only", choices=["ollama", "llamacpp"], default=None)
    args = ap.parse_args()

    ngl_values = args.ngl or ([0, 16, 33] if args.quick else DEFAULT_NGL)
    repeats = 1 if args.quick else args.repeats

    if not wait_http(f"{OLLAMA_URL}/api/tags", timeout=10):
        raise SystemExit("Ollama is not running. Start it with: ollama serve")

    gguf = gguf_path_for(args.model)
    RESULTS.mkdir(parents=True, exist_ok=True)

    print(f"model   {args.model}")
    print(f"weights {gguf}  ({gguf.stat().st_size / 1024**3:.2f} GiB)")
    print(f"sweep   ngl={ngl_values}  repeats={repeats}  max_tokens={args.max_tokens}\n")

    samples: list[Sample] = []
    if args.only != "ollama":
        samples += sweep_llamacpp(gguf, args.model, ngl_values, repeats, args.max_tokens)
    if args.only != "llamacpp":
        samples += sweep_ollama(args.model, ngl_values, repeats, args.max_tokens)

    payload = {
        "meta": {
            "model": args.model,
            "gguf": str(gguf),
            "gguf_bytes": gguf.stat().st_size,
            "ngl_values": ngl_values,
            "repeats": repeats,
            "max_tokens": args.max_tokens,
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "host": host_info(),
        },
        "summary": summarize(samples),
        "samples": [s.as_dict() for s in samples],
    }

    out = RESULTS / f"bench-{time.strftime('%Y%m%d-%H%M%S')}.json"
    out.write_text(json.dumps(payload, indent=2))
    latest = RESULTS / "latest.json"
    latest.write_text(json.dumps(payload, indent=2))

    print(f"\nwrote {out}")
    print(f"      {latest}")
    print("\nnext: python -m bench.chart")


if __name__ == "__main__":
    main()
