#!/usr/bin/env python3
"""Audit automatic NUMA balancing policy, activity, and Laguna provenance."""
import json
import re
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ENV = ROOT / "env.sh"
HARNESS = ROOT / "scripts/bench-serial.sh"
OUT = ROOT / "results/kernel-automatic-numa-balancing-audit-20260807.json"


def read(path: str) -> str | None:
    try:
        return Path(path).read_text().strip()
    except (FileNotFoundError, PermissionError):
        return None


def vmstat() -> dict[str, int]:
    values = {}
    for line in Path("/proc/vmstat").read_text().splitlines():
        name, value = line.split()
        if name.startswith("numa_"):
            values[name] = int(value)
    return values


raw = read("/proc/sys/kernel/numa_balancing")
if raw is None:
    raise RuntimeError("kernel does not export kernel.numa_balancing")
policy = int(raw)
rate_raw = read("/proc/sys/kernel/numa_balancing_promote_rate_limit_MBps")
env_text = ENV.read_text(errors="replace")
harness_text = HARNESS.read_text(errors="replace")
source = env_text + "\n" + harness_text
control_pattern = re.compile(r"numa_balancing|numactl|numa[_ -](?:balance|bind|policy)", re.I)
metric_pattern = re.compile(r"numa_(?:hint_faults|hint_faults_local|pages_migrated|pte_updates|huge_pte_updates)", re.I)
activity = vmstat()
report = {
    "angle": "kernel-automatic-numa-balancing",
    "generated_at": datetime.now(timezone.utc).isoformat(),
    "live": {
        "numa_balancing": policy,
        "automatic_numa_balancing_enabled": policy != 0,
        "promote_rate_limit_MBps": int(rate_raw) if rate_raw is not None else None,
        "vmstat_numa_activity": activity,
    },
    "laguna": {
        "env_or_harness_controls_numa_policy": bool(control_pattern.search(source)),
        "serial_harness_records_numa_activity": bool(metric_pattern.search(harness_text)),
    },
    "interpretation": (
        "Automatic NUMA balancing can introduce hint faults and page migrations. This audit records "
        "the live policy and counters and whether Laguna controls or records them."
    ),
}
assert policy >= 0
assert "numa_hint_faults" in activity
assert "numa_pages_migrated" in activity
OUT.parent.mkdir(parents=True, exist_ok=True)
OUT.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
print(OUT)
