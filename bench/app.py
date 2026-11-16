"""Streamlit dashboard for the sweep results.

    .venv/bin/streamlit run bench/app.py

Reads the JSON that `bench.run` writes. Unlike the stdlib dashboard in
`bench.serve`, this one lets you page back through historical sweeps and filter
interactively -- the things Streamlit is actually good at.
"""
from __future__ import annotations

import json
from pathlib import Path

import altair as alt
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results"

# Validated categorical slots 1 and 2. Light and dark steps were each checked
# against their own surface; see references/palette.md.
PALETTE = {
    "light": {"llama.cpp": "#2a78d6", "ollama": "#eb6834", "text2": "#52514e", "grid": "#e3e2de"},
    "dark": {"llama.cpp": "#3987e5", "ollama": "#d95926", "text2": "#c3c2b7", "grid": "#35342f"},
}
# Plain-language explanations surfaced on hover. Written for someone who has not
# read the README -- no jargon without unpacking it in the same breath.
HELP = {
    "prefill_metric":
        "How much faster the GPU gets through *reading your prompt*, versus doing it on "
        "the CPU alone. This step is called prefill: before the model writes a single "
        "word, it has to digest everything you sent it. It happens once per request.",
    "decode_metric":
        "How much faster the GPU *writes the answer*, versus the CPU alone. This step is "
        "called decode: the model produces the reply one token (roughly, one word-piece) "
        "at a time. It gains less from the GPU than reading does -- see the right-hand chart.",
    "fastest_metric":
        "The quickest answer-writing speed anywhere in this sweep, and the setup that "
        "produced it. As a rough feel: 40 tokens/second is noticeably faster than most "
        "people read.",
    "prefill_chart":
        "Moving left to right, more of the model is loaded onto the GPU. Higher is better. "
        "This line climbs steeply because reading a long prompt is a big pile of maths that "
        "can all happen at once -- precisely what a GPU is built for.",
    "decode_chart":
        "Same left-to-right meaning, but notice how much flatter this line is. Writing "
        "tokens is limited by how fast the model's weights can be pulled out of memory, "
        "not by raw calculation. On Apple Silicon the CPU and GPU share the same fast "
        "memory, so handing the work to the GPU helps far less here than it does for reading.",
    "table":
        "Every number behind the charts -- one row per engine, GPU-layer setting, and "
        "prompt. Click a column header to sort.",
    "run_picker":
        "Each entry is one benchmark sweep, labelled with when it ran. Pick an older one "
        "to see how results have changed.",
    "prompt_picker":
        "Which test prompt to display. 'long' is about 3,000 tokens, so most of the time "
        "goes into reading it. 'short' is a single sentence, so nearly all the time is "
        "spent writing the answer instead.",
    "engine_picker":
        "Which inference servers to show. Both ran the exact same model file, so any gap "
        "between them is the server software, not the model.",
    "ngl":
        "How many of the model's layers were placed on the GPU. 0 means the CPU did "
        "everything; 33 means the whole model ran on the GPU. The in-between values split "
        "the work.",
    "ttft": "Time to first token -- how long you stare at nothing before the first word appears.",
    "prefill_col": "How fast the model read the prompt, in tokens per second. Higher is better.",
    "decode_col": "How fast the model wrote the answer, in tokens per second. Higher is better.",
    "prompt_tok": "How many tokens the prompt came to once the model split it up.",
    "output_tok": "How many tokens the model generated before stopping.",
}

BACKENDS = ["llama.cpp", "ollama"]  # fixed order -- color follows the entity, never its rank


def theme_mode() -> str:
    """Follow the viewer's Streamlit theme when the running version exposes it."""
    ctx_theme = getattr(st.context, "theme", None)
    if ctx_theme is not None and getattr(ctx_theme, "type", None) in ("light", "dark"):
        return ctx_theme.type
    return "dark" if st.get_option("theme.base") == "dark" else "light"


def run_files() -> list[Path]:
    """Newest first; `latest.json` is a duplicate of one of these, so skip it."""
    return sorted(RESULTS.glob("bench-*.json"), reverse=True)


def load(path: Path) -> dict:
    return json.loads(path.read_text())


def chart(df: pd.DataFrame, field: str, colors: dict) -> alt.LayerChart:
    """One measure, one y-axis. Never two scales in a single frame."""
    present = [b for b in BACKENDS if b in set(df["backend"])]
    scale = alt.Scale(domain=present, range=[colors[b] for b in present])

    base = alt.Chart(df).encode(
        x=alt.X("ngl:Q",
                title="transformer layers offloaded to GPU",
                axis=alt.Axis(values=sorted(df["ngl"].unique()), grid=False,
                              labelColor=colors["text2"], titleColor=colors["text2"])),
        y=alt.Y(f"{field}:Q",
                title="tokens / second",
                scale=alt.Scale(zero=True, nice=True),
                axis=alt.Axis(gridColor=colors["grid"], domain=False, tickSize=0,
                              labelColor=colors["text2"], titleColor=colors["text2"])),
        color=alt.Color("backend:N", scale=scale,
                        legend=alt.Legend(title=None, orient="top-left",
                                          labelColor=colors["text2"])),
    )

    line = base.mark_line(strokeWidth=2, interpolate="linear")
    point = base.mark_point(size=80, filled=True, opacity=1)

    # Direct labels so identity never rests on color alone. They wear the muted
    # text token, not the series color.
    last = df.sort_values("ngl").groupby("backend", as_index=False).last()
    labels = alt.Chart(last).mark_text(
        align="left", dx=9, dy=0, fontSize=11, color=colors["text2"]
    ).encode(x="ngl:Q", y=f"{field}:Q", text="backend:N")

    hover = base.mark_circle(size=220, opacity=0).encode(
        tooltip=[
            alt.Tooltip("backend:N", title="engine"),
            alt.Tooltip("ngl:Q", title="GPU layers"),
            alt.Tooltip("prefill_tps:Q", title="prefill tok/s", format=".1f"),
            alt.Tooltip("decode_tps:Q", title="decode tok/s", format=".2f"),
            alt.Tooltip("ttft_s:Q", title="TTFT (s)", format=".3f"),
            alt.Tooltip("prompt_tokens:Q", title="prompt tokens"),
        ]
    )

    return (
        (line + point + labels + hover)
        .properties(height=300)
        .configure_view(strokeWidth=0)
        .configure_axis(labelFontSize=11, titleFontSize=11)
    )


def speedup(df: pd.DataFrame, field: str):
    """Best CPU-only -> full-offload ratio across the engines present."""
    best = None
    for backend in BACKENDS:
        pts = df[df["backend"] == backend].sort_values("ngl")
        if len(pts) < 2:
            continue
        lo, hi = pts.iloc[0], pts.iloc[-1]
        if lo[field] <= 0:
            continue
        ratio = hi[field] / lo[field]
        if best is None or ratio > best["ratio"]:
            best = {"ratio": ratio, "backend": backend, "lo": lo[field], "hi": hi[field],
                    "lo_ngl": int(lo["ngl"]), "hi_ngl": int(hi["ngl"])}
    return best


def main() -> None:
    st.set_page_config(page_title="local-llm-bench", page_icon="🔬", layout="wide")
    colors = PALETTE[theme_mode()]

    files = run_files()
    if not files:
        st.title("local-llm-bench")
        st.warning("No sweeps yet.")
        st.code(".venv/bin/python -m bench.run --quick", language="bash")
        st.stop()

    with st.sidebar:
        st.header("Sweep")
        chosen = st.selectbox("Run", files, help=HELP["run_picker"],
                              format_func=lambda p: p.stem.replace("bench-", ""))
        if st.button("Reload", width="stretch"):
            st.rerun()
        st.divider()
        prompt = st.radio("Prompt", ["long", "short"], help=HELP["prompt_picker"])
        backends = st.multiselect("Engines", BACKENDS, default=BACKENDS,
                                  help=HELP["engine_picker"])

    payload = load(chosen)
    meta, host = payload["meta"], payload["meta"].get("host", {})
    df = pd.DataFrame(payload["summary"])
    view = df[(df["prompt"] == prompt) & (df["backend"].isin(backends))]

    st.title("local-llm-bench")
    st.caption(
        f"`{meta['model']}` · {host.get('chip', '?')} · {host.get('ram_gb', '?')} GB · "
        f"median of {meta['repeats']} run(s) · {meta['timestamp']}"
    )

    with st.expander("What am I looking at?"):
        st.markdown(
            "Two programs can run the same AI model on this Mac: **llama.cpp** and "
            "**ollama**. This page runs both on the *identical* model file and times them.\n\n"
            "The knob being swept is how much of the model sits on the **GPU** rather than "
            "the CPU — `0` layers means the CPU does all the work, `33` means the GPU does. "
            "Every chart reads left-to-right as *more GPU*.\n\n"
            "Each request splits into two phases, and they behave differently:\n\n"
            "- **Prefill** — the model reads your prompt. Heavy parallel maths, so the GPU "
            "is dramatically better at it.\n"
            "- **Decode** — the model writes its reply, one token at a time. Limited by "
            "memory speed rather than calculation, so the GPU helps much less.\n\n"
            "Hover the ⓘ next to any heading, or any point on a chart, for detail."
        )

    if view.empty:
        st.info("No rows for that filter — pick at least one engine.")
        st.stop()

    pre, dec = speedup(view, "prefill_tps"), speedup(view, "decode_tps")
    fastest = view.loc[view["decode_tps"].idxmax()]

    c1, c2, c3 = st.columns(3)
    c1.metric(
        "Prefill: CPU → GPU",
        f"{pre['ratio']:.1f}×" if pre else "—",
        f"{pre['lo']:.0f} → {pre['hi']:.0f} tok/s ({pre['backend']})" if pre else None,
        delta_color="off",
        help=HELP["prefill_metric"],
    )
    c2.metric(
        "Decode: CPU → GPU",
        f"{dec['ratio']:.1f}×" if dec else "—",
        f"{dec['lo']:.1f} → {dec['hi']:.1f} tok/s ({dec['backend']})" if dec else None,
        delta_color="off",
        help=HELP["decode_metric"],
    )
    c3.metric(
        "Fastest decode",
        f"{fastest['decode_tps']:.1f} tok/s",
        f"{fastest['backend']} at {int(fastest['ngl'])} layers",
        delta_color="off",
        help=HELP["fastest_metric"],
    )

    left, right = st.columns(2)
    with left:
        st.subheader("Prefill throughput", help=HELP["prefill_chart"])
        st.caption("Reading the prompt. Hover any point for its exact numbers.")
        st.altair_chart(chart(view, "prefill_tps", colors), width="stretch")
    with right:
        st.subheader("Decode throughput", help=HELP["decode_chart"])
        st.caption("Writing the answer. Hover any point for its exact numbers.")
        st.altair_chart(chart(view, "decode_tps", colors), width="stretch")

    st.subheader("All measurements", help=HELP["table"])
    table = view[["backend", "ngl", "prompt", "ttft_s", "prefill_tps",
                  "decode_tps", "prompt_tokens", "completion_tokens"]]
    st.dataframe(
        table.sort_values(["backend", "ngl"]),
        hide_index=True,
        width="stretch",
        column_config={
            "ngl": st.column_config.NumberColumn("GPU layers", help=HELP["ngl"]),
            "ttft_s": st.column_config.NumberColumn("TTFT (s)", format="%.3f", help=HELP["ttft"]),
            "prefill_tps": st.column_config.NumberColumn("Prefill (tok/s)", format="%.1f",
                                                         help=HELP["prefill_col"]),
            "decode_tps": st.column_config.NumberColumn("Decode (tok/s)", format="%.2f",
                                                        help=HELP["decode_col"]),
            "prompt_tokens": st.column_config.NumberColumn("Prompt tok", help=HELP["prompt_tok"]),
            "completion_tokens": st.column_config.NumberColumn("Output tok", help=HELP["output_tok"]),
        },
    )

    with st.expander("Run metadata"):
        st.caption(f"Both engines served the same GGUF: `{meta.get('gguf', '')}`")
        st.json(meta)


if __name__ == "__main__":
    main()
