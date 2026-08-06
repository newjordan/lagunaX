#!/usr/bin/env python3
"""Measure the retired diagnostic check removed from Laguna's fused decode path."""

import json
import pathlib
import statistics
import subprocess
import tempfile
from datetime import date

ROOT = pathlib.Path(__file__).resolve().parents[1]
OUT = ROOT / "benchmark/results/fuse-trace-hotpath-20260807.json"
SOURCE = r'''
#include <atomic>
#include <chrono>
#include <cstdio>

static std::atomic<int> once{1};
static volatile int sink;

template <bool traced> double run() {
    constexpr int n = 100000000;
    auto start = std::chrono::steady_clock::now();
    for (int i = 0; i < n; ++i) {
        if constexpr (traced) {
            if (once.load(std::memory_order_relaxed) == 0 &&
                once.exchange(1, std::memory_order_relaxed) == 0) sink++;
        }
        asm volatile("" ::: "memory");
    }
    return std::chrono::duration<double, std::nano>(
        std::chrono::steady_clock::now() - start).count() / n;
}
int main() {
    for (int i = 0; i < 9; ++i) std::printf("%.9f %.9f\n", run<true>(), run<false>());
}
'''

with tempfile.TemporaryDirectory() as td:
    src = pathlib.Path(td) / "bench.cpp"
    exe = pathlib.Path(td) / "bench"
    src.write_text(SOURCE)
    subprocess.run(["c++", "-O3", "-std=c++17", str(src), "-o", str(exe)], check=True)
    rows = [tuple(map(float, line.split())) for line in subprocess.check_output([str(exe)], text=True).splitlines()]

old = [r[0] for r in rows]
new = [r[1] for r in rows]
result = {
    "date": str(date.today()),
    "iterations_per_trial": 100000000,
    "trials": len(rows),
    "old_atomic_check_ns_median": statistics.median(old),
    "new_trace_disabled_ns_median": statistics.median(new),
    "removed_ns_per_fused_op": statistics.median(old) - statistics.median(new),
    "speedup_of_diagnostic_check": statistics.median(old) / statistics.median(new),
    "scope": "synthetic host control-path benchmark; not end-to-end inference",
}
OUT.parent.mkdir(parents=True, exist_ok=True)
OUT.write_text(json.dumps(result, indent=2) + "\n")
print(OUT)
