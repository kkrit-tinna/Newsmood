"""`newsmood score --file ... --benchmark` (T2.5): throughput and latency.

Three numbers, measured differently on purpose:
  throughput      headlines / wall-clock seconds over the whole file, after
                  one untimed warmup batch (first-call allocation and kernel
                  selection are one-off costs, not steady state)
  batched p50/p95 each batch's time / its size, assigned to every headline in
                  that batch; percentiles over headlines
  cold latency    a fresh `newsmood score --text` process with the model
                  already cached: interpreter start, imports, model load and
                  one prediction. What a user typing the command waits for.
"""

from __future__ import annotations

import platform
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from newsmood.models.scoring import BatchClassifier

COLD_TEXT = "Profit fell sharply"
COLD_RUNS = 3


@dataclass(frozen=True)
class BenchmarkResult:
    device: str
    torch_version: str
    torch_threads: int
    chip: str
    n_headlines: int
    batch_size: int
    n_batches: int
    total_seconds: float
    p50_ms: float
    p95_ms: float
    cold_seconds: list[float]

    @property
    def throughput(self) -> float:
        return self.n_headlines / self.total_seconds


def chip() -> str:
    if sys.platform == "darwin":
        out = subprocess.run(["sysctl", "-n", "machdep.cpu.brand_string"], capture_output=True, text=True)
        if out.returncode == 0 and out.stdout.strip():
            return out.stdout.strip()
    return platform.processor() or platform.machine()


def time_batches(texts: list[str], classify: BatchClassifier, batch_size: int) -> tuple[float, list[float]]:
    """(total wall seconds, per-headline seconds for each headline), after one
    untimed warmup batch."""
    classify(texts[:batch_size])
    per_headline: list[float] = []
    total_start = time.perf_counter()
    for start in range(0, len(texts), batch_size):
        batch = texts[start : start + batch_size]
        t0 = time.perf_counter()
        classify(batch)
        elapsed = time.perf_counter() - t0
        per_headline += [elapsed / len(batch)] * len(batch)
    return time.perf_counter() - total_start, per_headline


def cold_command(device: str) -> list[str]:
    """The installed entry point next to this interpreter, so the timed
    process is the one a user runs; `python -m` if it is missing."""
    exe = Path(sys.executable).with_name("newsmood")
    base = [str(exe)] if exe.exists() else [sys.executable, "-m", "newsmood.cli"]
    return base + ["score", "--text", COLD_TEXT, "--device", device]


def time_cold(cmd: list[str], runs: int = COLD_RUNS) -> list[float]:
    times = []
    for _ in range(runs):
        t0 = time.perf_counter()
        out = subprocess.run(cmd, capture_output=True, text=True)
        times.append(time.perf_counter() - t0)
        if out.returncode != 0:
            raise RuntimeError(f"cold-latency run failed ({' '.join(cmd)}):\n{out.stderr}")
    return times


def run_benchmark(texts: list[str], classify: BatchClassifier, batch_size: int, device: str) -> BenchmarkResult:
    import torch

    if not texts:
        raise ValueError("benchmark needs at least one headline")
    total, per_headline = time_batches(texts, classify, batch_size)
    return BenchmarkResult(
        device=device,
        torch_version=torch.__version__,
        torch_threads=torch.get_num_threads(),
        chip=chip(),
        n_headlines=len(texts),
        batch_size=batch_size,
        n_batches=-(-len(texts) // batch_size),
        total_seconds=total,
        p50_ms=float(np.percentile(per_headline, 50)) * 1000,
        p95_ms=float(np.percentile(per_headline, 95)) * 1000,
        cold_seconds=time_cold(cold_command(device)),
    )


def format_benchmark(r: BenchmarkResult) -> str:
    cold = sorted(r.cold_seconds)
    return "\n".join(
        [
            f"device {r.device} · torch {r.torch_version} · {r.torch_threads} threads · {r.chip}",
            f"{r.n_headlines} headlines, batch size {r.batch_size} ({r.n_batches} batches), one warmup batch excluded",
            "",
            f"throughput        {r.throughput:,.1f} headlines/s  ({r.total_seconds:.2f} s total)",
            f"batched latency   p50 {r.p50_ms:.2f} ms · p95 {r.p95_ms:.2f} ms per headline  (batch time / batch size)",
            f"cold latency      {np.median(cold):.2f} s median of {len(cold)} (range {cold[0]:.2f}–{cold[-1]:.2f} s)",
            "                  fresh `newsmood score --text` process, model already cached:",
            "                  includes interpreter start, imports and model load, not just inference",
        ]
    )
