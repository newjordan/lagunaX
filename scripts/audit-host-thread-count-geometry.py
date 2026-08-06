#!/usr/bin/env python3
"""Audit benchmark thread-count geometry against physical/logical CPU topology."""
import json
import os
import re
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
env_text = (ROOT / "env.sh").read_text()
bench_text = (ROOT / "scripts/bench-serial.sh").read_text()

rows = []
for line in subprocess.check_output(
    ["lscpu", "-p=CPU,CORE,SOCKET,NODE"], text=True
).splitlines():
    if line and not line.startswith("#"):
        cpu, core, socket, node = map(int, line.split(","))
        rows.append({"cpu": cpu, "core": core, "socket": socket, "node": node})

logical = len(rows)
physical = len({(r["socket"], r["core"]) for r in rows})
sockets = len({r["socket"] for r in rows})
nodes = len({r["node"] for r in rows})
threads_match = re.search(r'^export THREADS="\$\{THREADS:-([0-9]+)\}"', env_text, re.M)
if not threads_match:
    raise SystemExit("THREADS default not found")
threads = int(os.environ.get("THREADS", threads_match.group(1)))

payload = {
    "audit": "host-thread-count-geometry",
    "topology": {
        "logical_cpus": logical,
        "physical_cores": physical,
        "threads_per_core": logical / physical,
        "sockets": sockets,
        "numa_nodes": nodes,
    },
    "policy": {
        "env_default_threads": int(threads_match.group(1)),
        "effective_threads": threads,
        "serial_harness_passes_threads": bool(re.search(r'-t\s+"\$THREADS"', bench_text)),
        "equals_physical_core_count": threads == physical,
        "uses_all_logical_cpus": threads == logical,
        "explicit_batch_thread_count": bool(re.search(r'--threads-batch|-tb\b', bench_text)),
    },
}

out = ROOT / "results" / "host-thread-count-geometry-audit-20260807.json"
out.write_text(json.dumps(payload, indent=2) + "\n")
print(out)
print(json.dumps(payload, indent=2))
