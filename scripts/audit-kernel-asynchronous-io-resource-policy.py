#!/usr/bin/env python3
"""Audit Linux asynchronous-I/O capacity/usage and Laguna provenance."""
import json
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ENV = (ROOT / "env.sh").read_text(errors="replace")
BENCH = (ROOT / "scripts/bench-serial.sh").read_text(errors="replace")
SYSCTLS = {
    "aio_max_nr": Path("/proc/sys/fs/aio-max-nr"),
    "aio_nr": Path("/proc/sys/fs/aio-nr"),
}

def read_int(path: Path) -> int:
    return int(path.read_text(errors="replace").strip())

live = {name: read_int(path) for name, path in SYSCTLS.items()}
terms = ("aio-max-nr", "aio-nr", "aio_max_nr", "aio_nr", "io_uring")
source = ENV + "\n" + BENCH
report = {
    "generated_utc": datetime.now(timezone.utc).isoformat(),
    "angle": "kernel-asynchronous-io-resource-policy",
    "live_policy": live,
    "aio_capacity_headroom": live["aio_max_nr"] - live["aio_nr"],
    "canonical_policy": {
        "env_or_harness_controls_async_io": [term for term in terms if term in source],
        "metrics_record_async_io": [term for term in terms if f'\"{term}\"' in BENCH],
    },
}
assert live["aio_max_nr"] > 0
assert 0 <= live["aio_nr"] <= live["aio_max_nr"]
assert not report["canonical_policy"]["env_or_harness_controls_async_io"]
assert not report["canonical_policy"]["metrics_record_async_io"]
out = ROOT / "results/kernel-asynchronous-io-resource-policy-audit-20260807.json"
out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
print(out)
