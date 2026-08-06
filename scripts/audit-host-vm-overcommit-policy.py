#!/usr/bin/env python3
"""Audit host virtual-memory overcommit policy and Laguna provenance."""
import json
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ENV = (ROOT / "env.sh").read_text(errors="replace")
BENCH = (ROOT / "scripts/bench-serial.sh").read_text(errors="replace")


def read(path):
    try:
        return Path(path).read_text(errors="replace").strip()
    except OSError:
        return None


fields = {
    "overcommit_memory": "/proc/sys/vm/overcommit_memory",
    "overcommit_ratio": "/proc/sys/vm/overcommit_ratio",
    "overcommit_kbytes": "/proc/sys/vm/overcommit_kbytes",
    "user_reserve_kbytes": "/proc/sys/vm/user_reserve_kbytes",
    "admin_reserve_kbytes": "/proc/sys/vm/admin_reserve_kbytes",
    "max_map_count": "/proc/sys/vm/max_map_count",
    "mmap_min_addr": "/proc/sys/vm/mmap_min_addr",
}
live = {name: read(path) for name, path in fields.items()}
meminfo = {}
for line in Path("/proc/meminfo").read_text().splitlines():
    key, value = line.split(":", 1)
    if key in {"MemTotal", "CommitLimit", "Committed_AS"}:
        meminfo[key] = value.strip()
source = ENV + "\n" + BENCH
terms = tuple(fields)
report = {
    "generated_utc": datetime.now(timezone.utc).isoformat(),
    "angle": "host-vm-overcommit-policy",
    "live_policy": live,
    "live_commit_accounting": meminfo,
    "canonical_policy": {
        "env_or_harness_controls_policy": [t for t in terms if t in source],
        "metrics_record_policy": [t for t in terms if f'"{t}"' in BENCH],
        "metrics_record_commit_accounting": [t for t in meminfo if f'"{t}"' in BENCH],
    },
}
assert all(value is not None for value in live.values())
assert {"MemTotal", "CommitLimit", "Committed_AS"} == set(meminfo)
assert not report["canonical_policy"]["env_or_harness_controls_policy"]
assert not report["canonical_policy"]["metrics_record_policy"]
assert not report["canonical_policy"]["metrics_record_commit_accounting"]
out = ROOT / "results/host-vm-overcommit-policy-audit-20260807.json"
out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
print(out)
