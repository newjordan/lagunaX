#!/usr/bin/env python3
"""Audit kernel perf-event policy and concurrent profiler contention provenance."""
import json
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ENV = (ROOT / "env.sh").read_text(errors="replace")
BENCH = (ROOT / "scripts/bench-serial.sh").read_text(errors="replace")

CONTROLS = (
    "perf_event_paranoid",
    "perf_event_max_sample_rate",
    "perf_cpu_time_max_percent",
    "perf_event_mlock_kb",
    "kptr_restrict",
)
PROFILER_NAMES = {
    "perf", "bpftrace", "bcc", "sysprof", "hotspot", "vtune", "amplxe-cl",
    "likwid-perfctr", "ocperf.py", "py-spy",
}


def read_sysctl(name):
    path = Path("/proc/sys/kernel") / name
    try:
        return path.read_text().strip()
    except OSError:
        return None


def active_profilers():
    found = []
    for proc in Path("/proc").iterdir():
        if not proc.name.isdigit():
            continue
        try:
            comm = (proc / "comm").read_text(errors="replace").strip()
            cmdline = (proc / "cmdline").read_bytes().replace(b"\0", b" ").decode(errors="replace").strip()
        except (OSError, PermissionError):
            continue
        executable = Path(cmdline.split()[0]).name if cmdline else comm
        if comm in PROFILER_NAMES or executable in PROFILER_NAMES:
            found.append({"pid": int(proc.name), "comm": comm, "cmdline": cmdline})
    return sorted(found, key=lambda item: item["pid"])


source = ENV + "\n" + BENCH
live = {name: read_sysctl(name) for name in CONTROLS}
report = {
    "generated_utc": datetime.now(timezone.utc).isoformat(),
    "angle": "kernel-perf-event-sampling-and-profiler-contention",
    "live_kernel_policy": live,
    "active_profiler_processes": active_profilers(),
    "canonical_policy": {
        "env_or_harness_controls_perf_policy": [name for name in CONTROLS if name in source],
        "harness_rejects_or_records_active_profilers": any(name in source for name in PROFILER_NAMES),
        "metrics_record_perf_policy": any(f'\"{name}\"' in BENCH for name in CONTROLS),
    },
}

assert all(value is not None for value in live.values())
assert not report["canonical_policy"]["env_or_harness_controls_perf_policy"]
assert not report["canonical_policy"]["harness_rejects_or_records_active_profilers"]
assert not report["canonical_policy"]["metrics_record_perf_policy"]
out = ROOT / "results/kernel-perf-event-profiler-contention-audit-20260807.json"
out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
print(out)
