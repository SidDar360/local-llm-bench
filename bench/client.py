"""The measurement driver: one OpenAI client, two base URLs.

Everything goes through the stock `openai` SDK. Switching engines is a change of
`base_url` and nothing else -- that is the point of the exercise.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, asdict

from openai import OpenAI


@dataclass
class Sample:
    backend: str
    ngl: int
    prompt_name: str
    ttft_s: float           # time to first streamed content token
    total_s: float          # wall clock for the whole request
    decode_s: float         # total minus prefill; the generation phase
    prompt_tokens: int      # tokens actually ingested -- guards against cache hits
    completion_tokens: int
    prefill_tps: float      # prompt_tokens / ttft_s -- the compute-bound half
    decode_tps: float       # completion_tokens / decode_s -- the bandwidth-bound half
    token_source: str       # "usage" if the server reported it, else "chunks"

    def as_dict(self) -> dict:
        return asdict(self)


def client_for(base_url: str) -> OpenAI:
    # Local servers ignore the key, but the SDK requires one to be present.
    return OpenAI(base_url=base_url, api_key="local-no-key-needed", timeout=600.0)


def measure(
    client: OpenAI,
    model: str,
    prompt: str,
    *,
    backend: str,
    ngl: int,
    prompt_name: str,
    max_tokens: int = 256,
) -> Sample:
    """Run one streamed completion and time the two phases separately.

    TTFT covers prompt ingestion (prefill); the remainder is token generation
    (decode). They scale differently with GPU offload, so a single
    tokens-per-second number would hide the interesting half.

    The caller is responsible for passing a prompt that has not been seen before
    -- see `run._unique`. Both engines serve a repeated prompt from the KV cache,
    which makes TTFT measure a cache hit rather than prefill.
    """
    t0 = time.perf_counter()
    ttft: float | None = None
    chunks = 0
    usage_completion: int | None = None
    usage_prompt: int | None = None

    stream = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=max_tokens,
        temperature=0.0,
        stream=True,
        stream_options={"include_usage": True},
    )
    for event in stream:
        if getattr(event, "usage", None) is not None:
            usage_completion = event.usage.completion_tokens
            usage_prompt = event.usage.prompt_tokens
        if not event.choices:
            continue
        delta = event.choices[0].delta
        if getattr(delta, "content", None):
            if ttft is None:
                ttft = time.perf_counter() - t0
            chunks += 1

    total = time.perf_counter() - t0
    if ttft is None:  # model emitted nothing; treat the whole call as prefill
        ttft = total

    if usage_completion:
        tokens, source = usage_completion, "usage"
    else:
        tokens, source = chunks, "chunks"

    decode = max(total - ttft, 1e-9)
    prompt_tokens = usage_prompt or 0
    return Sample(
        backend=backend,
        ngl=ngl,
        prompt_name=prompt_name,
        ttft_s=round(ttft, 4),
        total_s=round(total, 4),
        decode_s=round(decode, 4),
        prompt_tokens=prompt_tokens,
        completion_tokens=tokens,
        prefill_tps=round(prompt_tokens / max(ttft, 1e-9), 1),
        decode_tps=round(tokens / decode, 2),
        token_source=source,
    )
