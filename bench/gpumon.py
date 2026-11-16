"""Optional: sample Metal GPU utilization during a run.

`powermetrics` needs root, so this is deliberately kept out of the main sweep --
no sudo prompt in the middle of a benchmark. Run it in a second terminal:

    sudo python -m bench.gpumon --seconds 30

The -ngl sweep already demonstrates GPU usage through throughput; this is for
when you want the utilization number itself.
"""
from __future__ import annotations

import argparse
import re
import subprocess
import sys


def sample(seconds: int, interval_ms: int = 500) -> None:
    n = max(1, int(seconds * 1000 / interval_ms))
    cmd = ["powermetrics", "--samplers", "gpu_power", "-i", str(interval_ms), "-n", str(n)]
    try:
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True)
    except FileNotFoundError:
        sys.exit("powermetrics not found (macOS only)")

    busy_re = re.compile(r"GPU (?:HW )?active residency:\s+([\d.]+)%")
    power_re = re.compile(r"GPU Power:\s+(\d+)\s*mW")
    samples: list[float] = []
    assert proc.stdout is not None
    for line in proc.stdout:
        if m := busy_re.search(line):
            value = float(m.group(1))
            samples.append(value)
            print(f"GPU active {value:5.1f}%")
        elif m := power_re.search(line):
            print(f"GPU power  {int(m.group(1)):5d} mW")
    proc.wait()

    if samples:
        print(f"\nmean GPU active residency: {sum(samples) / len(samples):.1f}%  "
              f"(peak {max(samples):.1f}%)")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seconds", type=int, default=30)
    args = ap.parse_args()
    sample(args.seconds)


if __name__ == "__main__":
    main()
