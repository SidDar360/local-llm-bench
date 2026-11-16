"""Render the sweep results.

Two panels, one measure each -- never a shared axis with two scales. Decode
throughput is read off the short prompt (generation-bound); time-to-first-token
off the long prompt (prefill-bound), which is where offload matters most.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

RESULTS = Path(__file__).resolve().parent.parent / "results"

# Validated categorical slots 1 and 2 (light mode). See references/palette.md.
SERIES = {"llama.cpp": "#2a78d6", "ollama": "#eb6834"}
SURFACE = "#fcfcfb"
TEXT_PRIMARY = "#0b0b0b"
TEXT_SECONDARY = "#52514e"
GRID = "#e3e2de"


def _series(summary: list[dict], backend: str, prompt: str, field: str):
    rows = sorted(
        (r for r in summary if r["backend"] == backend and r["prompt"] == prompt),
        key=lambda r: r["ngl"],
    )
    return [r["ngl"] for r in rows], [r[field] for r in rows]


def _panel(ax, summary, prompt, field, title, ylabel):
    ax.set_facecolor(SURFACE)
    plotted = False
    for backend, color in SERIES.items():
        x, y = _series(summary, backend, prompt, field)
        if not x:
            continue
        plotted = True
        ax.plot(x, y, color=color, linewidth=2, marker="o", markersize=7,
                markeredgecolor=SURFACE, markeredgewidth=1.5, label=backend, zorder=3)
        # Direct label on the final point -- identity is never color-alone.
        ax.annotate(backend, (x[-1], y[-1]), textcoords="offset points",
                    xytext=(8, 0), va="center", fontsize=9, color=TEXT_SECONDARY)

    ax.set_title(title, fontsize=11, color=TEXT_PRIMARY, loc="left", pad=10)
    ax.set_xlabel("transformer layers offloaded to GPU", fontsize=9, color=TEXT_SECONDARY)
    ax.set_ylabel(ylabel, fontsize=9, color=TEXT_SECONDARY)
    ax.grid(axis="y", color=GRID, linewidth=1, zorder=0)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.tick_params(colors=TEXT_SECONDARY, labelsize=9, length=0)
    ax.set_ylim(bottom=0)
    ax.margins(x=0.18)
    if not plotted:
        ax.text(0.5, 0.5, "no data", ha="center", va="center",
                transform=ax.transAxes, color=TEXT_SECONDARY)


def render(payload: dict, out: Path) -> Path:
    summary = payload["summary"]
    meta = payload["meta"]
    host = meta.get("host", {})

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6), facecolor=SURFACE)

    _panel(axes[0], summary, "short", "decode_tps",
           "Generation throughput", "tokens / second")
    _panel(axes[1], summary, "long", "ttft_s",
           "Time to first token (long prompt)", "seconds")

    fig.suptitle(
        f"{meta['model']} on {host.get('chip', 'this machine')} "
        f"— same GGUF, two servers",
        fontsize=13, color=TEXT_PRIMARY, x=0.012, ha="left", y=0.98,
    )
    handles, labels = axes[0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="upper right", frameon=False,
                   fontsize=9, labelcolor=TEXT_SECONDARY, ncols=len(labels),
                   bbox_to_anchor=(0.99, 1.005))

    fig.text(0.012, 0.015,
             f"median of {meta['repeats']} run(s) · {meta['max_tokens']} max tokens · "
             f"{meta['timestamp']}",
             fontsize=8, color=TEXT_SECONDARY)

    fig.tight_layout(rect=(0, 0.04, 1, 0.93))
    fig.savefig(out, dpi=200, facecolor=SURFACE)
    plt.close(fig)
    return out


def print_table(payload: dict) -> None:
    """The table view -- the chart is never the only way to read the numbers."""
    print(f"\n{'backend':<10} {'ngl':>4} {'prompt':<6} {'ttft (s)':>9} {'tok/s':>8} {'tokens':>7}")
    print("-" * 50)
    for r in sorted(payload["summary"], key=lambda r: (r["prompt"], r["backend"], r["ngl"])):
        print(f"{r['backend']:<10} {r['ngl']:>4} {r['prompt']:<6} "
              f"{r['ttft_s']:>9.3f} {r['decode_tps']:>8.2f} {r['completion_tokens']:>7}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--input", type=Path, default=RESULTS / "latest.json")
    ap.add_argument("--out", type=Path, default=RESULTS / "benchmark.png")
    args = ap.parse_args()

    if not args.input.exists():
        raise SystemExit(f"no results at {args.input} — run: python -m bench.run")

    payload = json.loads(args.input.read_text())
    print_table(payload)
    out = render(payload, args.out)
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
