#!/usr/bin/env python3
"""Audit benchmark process resource limits and Laguna provenance."""
import json
import resource
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ENV = (ROOT / "env.sh").read_text(errors="replace")
BENCH = (ROOT / "scripts/bench-serial.sh").read_text(errors="replace")

LIMITS = {
    "address_space_bytes": resource.RLIMIT_AS,
    "core_file_bytes": resource.RLIMIT_CORE,
    "data_segment_bytes": resource.RLIMIT_DATA,
    "locked_memory_bytes": resource.RLIMIT_MEMLOCK,
    "open_files": resource.RLIMIT_NOFILE,
    "processes": resource.RLIMIT_NPROC,
    "resident_set_bytes": resource.RLIMIT_RSS,
    "stack_bytes": resource.RLIMIT_STACK,
}

def encode(value: int):
    return "unlimited" if value == resource.RLIM_INFINITY else value

live = {}
for name, kind in LIMITS.items():
    soft, hard = resource.getrlimit(kind)
    live[name] = {"soft": encode(soft), "hard": encode(hard)}

source = ENV + "\n" + BENCH
policy_tokens = ("ulimit", "prlimit", "setrlimit", "RLIMIT_")
report = {
    "angle": "benchmark-process-resource-limits",
    "generated_at": datetime.now(timezone.utc).isoformat(),
    "live": live,
    "laguna": {
        "env_policy_hits": [token for token in policy_tokens if token in ENV],
        "bench_policy_hits": [token for token in policy_tokens if token in BENCH],
        "metrics_record_limits": any(f'\"{name}\"' in BENCH for name in LIMITS),
    },
    "finding": (
        "Laguna neither configures nor records process resource limits, including "
        "locked-memory, open-file, stack, address-space, and core-file limits."
    ),
}
out = ROOT / "results/benchmark-process-resource-limits-audit-20260807.json"
out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
print(out)
