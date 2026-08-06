#!/usr/bin/env python3
"""Audit NUMA memory-placement policy for the active Laguna benchmark."""
from __future__ import annotations
import json, os, pathlib, re, subprocess

ROOT = pathlib.Path(__file__).resolve().parents[1]
ENV = ROOT / "env.sh"
BENCH = ROOT / "scripts" / "bench-serial.sh"
RESULTS = ROOT / "results"
sources = {str(p.relative_to(ROOT)): p.read_text(errors="replace") for p in (ENV, BENCH)}
bench_bin = os.environ.get("LX_LLAMA_BENCH")
if not bench_bin:
    probe = subprocess.run(
        ["bash", "-c", 'source "$1"; printf "%s" "$LX_LLAMA_BENCH"', "bash", str(ENV)],
        text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=30, check=True,
    )
    bench_bin = probe.stdout
help_text = subprocess.run(
    ["bash", "-c", 'source "$1"; exec "$2" --help', "bash", str(ENV), bench_bin],
    text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=30, check=False,
).stdout
help_line = next((line.strip() for line in help_text.splitlines() if "--numa" in line), None)
node_text = pathlib.Path("/sys/devices/system/node/online").read_text().strip()
mentions = []
for path in RESULTS.rglob("*.json"):
    if path.name.startswith("numa-memory-placement-audit-"):
        continue
    text = path.read_text(errors="replace")
    if re.search(r'(?i)(--numa|numa_mode|numactl)', text):
        mentions.append(str(path.relative_to(ROOT)))
report = {
    "policy": "numa_memory_placement",
    "llama_bench": bench_bin,
    "help_line": help_line,
    "executable_default": "disabled" if help_line and "default: disabled" in help_line else "unknown",
    "active_source_overrides": {name: bool(re.search(r'(?i)(--numa|numactl)', text)) for name, text in sources.items()},
    "process_environment_overrides": {key: value for key, value in os.environ.items() if "NUMA" in key.upper()},
    "host_numa_nodes_online": node_text,
    "historical_json_mentions": mentions,
    "historical_json_mention_count": len(mentions),
}
out = RESULTS / "numa-memory-placement-audit-20260807.json"
out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
print(out)
