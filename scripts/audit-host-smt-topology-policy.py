#!/usr/bin/env python3
"""Audit simultaneous multithreading topology and Laguna provenance."""
import json
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ENV = (ROOT / "env.sh").read_text(errors="replace")
BENCH = (ROOT / "scripts/bench-serial.sh").read_text(errors="replace")


def read(path):
    try:
        return path.read_text(errors="replace").strip()
    except OSError:
        return None


cpu_root = Path("/sys/devices/system/cpu")
cpus = sorted(cpu_root.glob("cpu[0-9]*"), key=lambda p: int(p.name[3:]))
siblings = {}
for cpu in cpus:
    online = read(cpu / "online")
    if online in (None, "1"):
        siblings[cpu.name] = read(cpu / "topology/thread_siblings_list")

source = ENV + "\n" + BENCH
control_terms = ("nosmt", "/sys/devices/system/cpu/smt/control", "thread_siblings", "SMT_CONTROL")
metric_terms = ("smt_active", "smt_control", "thread_siblings_list", "physical_core_count")
report = {
    "generated_utc": datetime.now(timezone.utc).isoformat(),
    "angle": "host-smt-topology-policy",
    "live_policy": {
        "smt_active": read(cpu_root / "smt/active"),
        "smt_control": read(cpu_root / "smt/control"),
        "online_logical_cpu_count": len(siblings),
        "thread_siblings_by_cpu": siblings,
        "unique_thread_sibling_groups": sorted(set(siblings.values())),
        "logical_cpus_per_core": sorted({len(v.split(",")) for v in siblings.values() if v}),
        "kernel_cmdline_nosmt": "nosmt" in (read(Path("/proc/cmdline")) or "").split(),
    },
    "canonical_policy": {
        "env_or_harness_controls_smt": [term for term in control_terms if term in source],
        "metrics_record_smt": [term for term in metric_terms if f'"{term}"' in BENCH],
    },
}
assert report["live_policy"]["smt_active"] in ("0", "1")
assert report["live_policy"]["smt_control"]
assert siblings and all(siblings.values())
assert not report["canonical_policy"]["env_or_harness_controls_smt"]
assert not report["canonical_policy"]["metrics_record_smt"]
out = ROOT / "results/host-smt-topology-policy-audit-20260807.json"
out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
print(out)
