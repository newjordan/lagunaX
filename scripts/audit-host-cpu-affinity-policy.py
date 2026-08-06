#!/usr/bin/env python3
"""Audit host CPU-affinity policy independently of thread-count geometry."""
import json
import os
import re
import subprocess
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ENV = ROOT / "env.sh"
HARNESS = ROOT / "scripts/bench-serial.sh"
RESULTS = ROOT / "results"

source = ENV.read_text() + "\n" + HARNESS.read_text()
controls = ["taskset", "numactl", "GOMP_CPU_AFFINITY", "KMP_AFFINITY", "OMP_PROC_BIND", "OMP_PLACES"]
mentions = {name: bool(re.search(rf"(?<![A-Za-z0-9_]){re.escape(name)}(?![A-Za-z0-9_])", source)) for name in controls}
status = Path("/proc/self/status").read_text()
allowed = re.search(r"^Cpus_allowed_list:\s*(.+)$", status, re.MULTILINE).group(1)
rows = subprocess.check_output(["lscpu", "-p=CPU,CORE,SOCKET,NODE,ONLINE"], text=True)
cpus = [line.split(",") for line in rows.splitlines() if line and not line.startswith("#")]
online = [row for row in cpus if row[4] == "Y"]
physical_cores = len({(row[2], row[1]) for row in online})
logical_cpus = len(online)
affinity = sorted(os.sched_getaffinity(0))
artifact = {
    "timestamp_utc": datetime.now(timezone.utc).isoformat(),
    "policy": "host-cpu-affinity",
    "allowed_cpu_list": allowed,
    "allowed_cpu_count": len(affinity),
    "online_logical_cpus": logical_cpus,
    "physical_cores": physical_cores,
    "smt_threads_per_core": logical_cpus // physical_cores,
    "active_source_controls": [name for name, found in mentions.items() if found],
    "control_mentions": mentions,
    "explicit_affinity_policy": any(mentions.values()),
}
RESULTS.mkdir(exist_ok=True)
out = RESULTS / "host-cpu-affinity-policy-audit-20260807.json"
out.write_text(json.dumps(artifact, indent=2) + "\n")
print(out)
print(json.dumps(artifact, indent=2))
