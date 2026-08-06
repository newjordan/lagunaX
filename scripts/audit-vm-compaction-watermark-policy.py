#!/usr/bin/env python3
"""Audit VM compaction, watermark, and reclaim-reserve policy for Laguna."""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "vm-compaction-watermark-policy-audit-20260807.json"


def read(path: Path) -> str | None:
    try:
        return path.read_text().strip()
    except OSError:
        return None


controls = {
    name: read(Path("/proc/sys/vm") / name)
    for name in (
        "compaction_proactiveness",
        "extfrag_threshold",
        "watermark_boost_factor",
        "watermark_scale_factor",
        "min_free_kbytes",
        "zone_reclaim_mode",
        "percpu_pagelist_high_fraction",
    )
}
vmstat_names = (
    "compact_stall",
    "compact_success",
    "compact_fail",
    "compact_migrate_scanned",
    "compact_free_scanned",
    "compact_daemon_wake",
    "compact_daemon_migrate_scanned",
    "compact_daemon_free_scanned",
)
vmstat: dict[str, int] = {}
for line in (read(Path("/proc/vmstat")) or "").splitlines():
    fields = line.split()
    if len(fields) == 2 and fields[0] in vmstat_names:
        vmstat[fields[0]] = int(fields[1])

env = (ROOT / "env.sh").read_text().lower()
harness = (ROOT / "scripts" / "bench-serial.sh").read_text().lower()
needles = tuple(controls) + ("compact_stall", "compact_success", "compact_fail")
report = {
    "generated_utc": datetime.now(timezone.utc).isoformat(),
    "scope": "VM compaction, fragmentation threshold, and free-memory watermark policy",
    "live": {"controls": controls, "vmstat_lifetime_counters": vmstat},
    "laguna": {
        "canonical_environment": "env.sh",
        "serial_harness": "scripts/bench-serial.sh",
        "configures_compaction_or_watermarks": any(n in env for n in needles),
        "records_compaction_or_watermarks": any(n in harness for n in needles),
    },
}
OUT.write_text(json.dumps(report, indent=2) + "\n")
print(OUT)
