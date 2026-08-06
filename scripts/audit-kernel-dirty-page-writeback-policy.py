#!/usr/bin/env python3
"""Audit kernel dirty-page writeback policy and Laguna benchmark provenance."""
import json
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ENV = (ROOT / "env.sh").read_text(errors="replace")
BENCH = (ROOT / "scripts/bench-serial.sh").read_text(errors="replace")

SYSCTLS = {
    "dirty_background_bytes": "/proc/sys/vm/dirty_background_bytes",
    "dirty_background_ratio": "/proc/sys/vm/dirty_background_ratio",
    "dirty_bytes": "/proc/sys/vm/dirty_bytes",
    "dirty_ratio": "/proc/sys/vm/dirty_ratio",
    "dirty_expire_centisecs": "/proc/sys/vm/dirty_expire_centisecs",
    "dirty_writeback_centisecs": "/proc/sys/vm/dirty_writeback_centisecs",
}


def read_int(path):
    try:
        return int(Path(path).read_text().strip())
    except (OSError, ValueError):
        return None


live = {name: read_int(path) for name, path in SYSCTLS.items()}
tokens = tuple(SYSCTLS)
report = {
    "angle": "kernel-dirty-page-writeback-policy",
    "generated_at": datetime.now(timezone.utc).isoformat(),
    "live": live,
    "laguna": {
        "env_policy_hits": [token for token in tokens if token in ENV],
        "bench_policy_hits": [token for token in tokens if token in BENCH],
        "metrics_record_policy": any(f'"{token}"' in BENCH for token in tokens),
    },
}
out = ROOT / "results/kernel-dirty-page-writeback-policy-audit-20260807.json"
out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
print(out)

assert all(value is not None for value in live.values())
assert not report["laguna"]["env_policy_hits"]
assert not report["laguna"]["bench_policy_hits"]
assert not report["laguna"]["metrics_record_policy"]
