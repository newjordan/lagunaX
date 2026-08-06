#!/usr/bin/env python3
"""Audit Laguna's host thread-count policy and provenance."""
import json, os, re, subprocess
from pathlib import Path

root = Path(__file__).resolve().parents[1]
env = (root / "env.sh").read_text()
harness = (root / "scripts/bench-serial.sh").read_text()
env_values = subprocess.run(
    ["bash", "-c", 'source "$1" >/dev/null; printf "%s\\n%s\\n" "$LX_LLAMA_BENCH" "$THREADS"', "audit", str(root / "env.sh")],
    text=True, capture_output=True, check=True,
).stdout.splitlines()
bench, configured_threads = env_values
help_text = subprocess.run(
    ["bash", "-c", 'source "$1" >/dev/null; "$LX_LLAMA_BENCH" --help 2>&1 || [[ $? == 1 ]]', "audit", str(root / "env.sh")],
    text=True, capture_output=True, check=True,
).stdout
lscpu = json.loads(subprocess.run(["lscpu", "-J"], text=True, capture_output=True, check=True).stdout)
cpu = {x["field"].rstrip(":"): x["data"] for x in lscpu["lscpu"]}
def integer(name):
    return int(cpu[name])
thread_default = re.search(r'--threads <n>\s+\(default: (\d+)\)', help_text)
configured = re.search(r'export THREADS=', env)
artifact = {
    "angle": "host_thread_count_policy",
    "host": {
        "logical_cpus": integer("CPU(s)"),
        "cores_per_socket": integer("Core(s) per socket"),
        "threads_per_core": integer("Thread(s) per core"),
        "sockets": integer("Socket(s)"),
    },
    "llama_bench": {
        "threads_supported": bool(thread_default),
        "threads_default": int(thread_default.group(1)) if thread_default else None,
    },
    "laguna": {
        "configured_threads": int(configured_threads) if configured else None,
        "passes_threads": '-t "$THREADS"' in harness,
        "records_threads_in_metrics": bool(re.search(r'"threads"\s*:', harness)),
    },
}
out = root / "results" / "host-thread-count-policy-audit-20260807.json"
out.write_text(json.dumps(artifact, indent=2) + "\n")
print(out)
