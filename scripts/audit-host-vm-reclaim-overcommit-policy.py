#!/usr/bin/env python3
"""Audit host virtual-memory reclaim/overcommit policy and Laguna provenance."""
import json
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ENV = (ROOT / "env.sh").read_text(errors="replace")
BENCH = (ROOT / "scripts/bench-serial.sh").read_text(errors="replace")

CONTROLS = (
    "overcommit_memory", "overcommit_ratio", "overcommit_kbytes",
    "zone_reclaim_mode", "watermark_scale_factor", "watermark_boost_factor",
    "compaction_proactiveness", "extfrag_threshold",
)

def read_sysctl(name):
    path = Path("/proc/sys/vm") / name
    try:
        return int(path.read_text().strip())
    except (OSError, ValueError):
        return None

def hits(text):
    return [name for name in CONTROLS if name in text]

live = {name: read_sysctl(name) for name in CONTROLS}
report = {
    "angle": "host-vm-reclaim-overcommit-policy",
    "generated_at": datetime.now(timezone.utc).isoformat(),
    "live": live,
    "laguna": {
        "env_policy_hits": hits(ENV),
        "bench_policy_hits": hits(BENCH),
        "metrics_record_policy": any(f'\"{name}\"' in BENCH for name in CONTROLS),
    },
    "finding": (
        "Laguna neither configures nor records the audited host virtual-memory "
        "overcommit, reclaim-watermark, or compaction policy."
    ),
}
out = ROOT / "results/host-vm-reclaim-overcommit-policy-audit-20260807.json"
out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
print(out)
