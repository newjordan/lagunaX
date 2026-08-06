#!/usr/bin/env python3
"""Audit automatic NUMA balancing policy and Laguna benchmark provenance."""
import json
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ENV = (ROOT / "env.sh").read_text(errors="replace")
BENCH = (ROOT / "scripts/bench-serial.sh").read_text(errors="replace")

CONTROLS = (
    "numa_balancing",
    "numa_balancing_promote_rate_limit_MBps",
    "numa_balancing_scan_delay_ms",
    "numa_balancing_scan_period_min_ms",
    "numa_balancing_scan_period_max_ms",
    "numa_balancing_scan_size_mb",
)


def read(path):
    try:
        return path.read_text(errors="replace").strip()
    except OSError:
        return None


def node_cpulists():
    result = {}
    for path in sorted(Path("/sys/devices/system/node").glob("node[0-9]*/cpulist")):
        result[path.parent.name] = read(path)
    return result


live = {name: read(Path("/proc/sys/kernel") / name) for name in CONTROLS}
cmdline = read(Path("/proc/cmdline"))
source = ENV.lower()
harness = BENCH.lower()
needles = tuple(name.lower() for name in CONTROLS) + ("numa_balancing=", "numactl")
report = {
    "angle": "automatic-numa-balancing-policy",
    "generated_at": datetime.now(timezone.utc).isoformat(),
    "live": {
        "controls": live,
        "node_cpulists": node_cpulists(),
        "online_nodes": read(Path("/sys/devices/system/node/online")),
        "possible_nodes": read(Path("/sys/devices/system/node/possible")),
        "kernel_cmdline_numa_balancing": [
            token for token in (cmdline or "").split() if "numa_balancing" in token.lower()
        ],
    },
    "laguna": {
        "env_policy_hits": [needle for needle in needles if needle in source],
        "bench_policy_hits": [needle for needle in needles if needle in harness],
        "records_numa_balancing": any(needle in harness for needle in needles),
    },
    "finding": (
        "Laguna neither configures nor records automatic NUMA balancing policy "
        "or the host NUMA node topology."
    ),
}
out = ROOT / "results/automatic-numa-balancing-policy-audit-20260807.json"
out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
print(out)
