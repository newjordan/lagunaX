#!/usr/bin/env python3
"""Audit benchmark cgroup CPU/memory constraints and their provenance."""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "benchmark-cgroup-resource-policy-audit-20260807.json"


def read(path: Path):
    try:
        return path.read_text().strip()
    except (OSError, UnicodeDecodeError):
        return None


def membership():
    rows = []
    for line in (read(Path("/proc/self/cgroup")) or "").splitlines():
        hierarchy, controllers, path = line.split(":", 2)
        rows.append({"hierarchy": hierarchy, "controllers": controllers, "path": path})
    return rows


members = membership()
cgroup_path = members[0]["path"].lstrip("/") if members else ""
cg = Path("/sys/fs/cgroup") / cgroup_path
fields = (
    "cgroup.controllers", "cgroup.subtree_control", "cpu.max", "cpu.weight",
    "cpuset.cpus", "cpuset.cpus.effective", "cpuset.mems", "cpuset.mems.effective",
    "memory.max", "memory.high", "memory.swap.max", "pids.max",
)
source = (ROOT / "env.sh").read_text() + "\n" + (ROOT / "scripts/bench-serial.sh").read_text()
needles = ("cpu.max", "cpu.weight", "cpuset.cpus", "cpuset.mems", "cgroup", "memory.max", "pids.max")
report = {
    "audit": "benchmark-cgroup-resource-policy",
    "live": {
        "mountinfo_cgroup2": [line for line in (read(Path("/proc/self/mountinfo")) or "").splitlines() if " - cgroup2 " in line],
        "membership": members,
        "effective_cgroup_path": str(cg),
        "effective_constraints": {name: read(cg / name) for name in fields},
    },
    "laguna": {
        "configured_controls": {name: (name in source) for name in needles},
        "metrics_record_policy": any(token in (ROOT / "scripts/bench-serial.sh").read_text() for token in ("cgroup_path", "cpu_max", "cpu_weight", "cpuset_cpus", "memory_max", "pids_max")),
    },
}
OUT.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
print(OUT)
