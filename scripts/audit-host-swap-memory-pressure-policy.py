#!/usr/bin/env python3
"""Audit host swap and memory-pressure policy used by Laguna benchmarks."""
import json
from pathlib import Path
from datetime import datetime, timezone

ROOT = Path(__file__).resolve().parents[1]
ENV = (ROOT / "env.sh").read_text()
BENCH = (ROOT / "scripts/bench-serial.sh").read_text()

def read(path):
    try:
        return Path(path).read_text().strip()
    except OSError:
        return None

meminfo = {}
for line in Path("/proc/meminfo").read_text().splitlines():
    if ":" in line:
        k, v = line.split(":", 1)
        meminfo[k] = v.strip()

swap_devices = []
lines = Path("/proc/swaps").read_text().splitlines()
for line in lines[1:]:
    fields = line.split()
    if len(fields) >= 5:
        swap_devices.append({"filename": fields[0], "type": fields[1], "size_kib": int(fields[2]), "used_kib": int(fields[3]), "priority": int(fields[4])})

needles = ("swappiness", "swapoff", "memory.high", "memory.max", "memory.swap.max", "oom_score_adj")
def hits(text):
    return [n for n in needles if n in text]

artifact = {
    "collected_at": datetime.now(timezone.utc).isoformat(),
    "live": {
        "vm_swappiness": int(read("/proc/sys/vm/swappiness")),
        "swap_total_kib": int(meminfo["SwapTotal"].split()[0]),
        "swap_free_kib": int(meminfo["SwapFree"].split()[0]),
        "mem_available_kib": int(meminfo["MemAvailable"].split()[0]),
        "swap_devices": swap_devices,
    },
    "laguna": {
        "env_policy_hits": hits(ENV),
        "bench_policy_hits": hits(BENCH),
        "records_memory_pressure_or_swap": any(n in BENCH for n in ("SwapTotal", "SwapFree", "MemAvailable", "vm_swappiness")),
    },
}
out = ROOT / "benchmark/results/host-swap-memory-pressure-policy-audit-20260807.json"
out.parent.mkdir(parents=True, exist_ok=True)
out.write_text(json.dumps(artifact, indent=2) + "\n")
print(out)
