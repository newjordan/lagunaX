#!/usr/bin/env python3
"""Audit llama-bench operation-offload configuration and provenance."""
import json
import os
import re
import subprocess
from pathlib import Path

root = Path(__file__).resolve().parents[1]
env_text = (root / "env.sh").read_text()
bench_text = (root / "scripts/bench-serial.sh").read_text()
env = os.environ.copy()
exec((root / "env.sh").read_text(), {}) if False else None
bench = env.get("LX_LLAMA_BENCH")
if not bench:
    match = re.search(r'export LX_BIN="\$\{LX_BIN:-([^}]+)\}"', env_text)
    if not match:
        raise SystemExit("cannot resolve LX_BIN")
    bench = str(Path(match.group(1)) / "llama-bench")
help_run = subprocess.run(
    ["bash", "-lc", 'source "$1" >/dev/null 2>&1; exec "$2" --help', "audit", str(root / "env.sh"), bench],
    text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
help_text = help_run.stdout
if help_run.returncode not in (0, 1):
    raise SystemExit(f"llama-bench --help failed ({help_run.returncode}): {help_text[:500]}")
m = re.search(r'--no-op-offload <0\|1>\s+\(default: ([01])\)', help_text)
if not m:
    raise SystemExit("no-op-offload contract not found")
common = bench_text.split("COMMON=(", 1)[1].split(")", 1)[0]
metrics = bench_text.split('"flags": {', 1)[1].split("}", 1)[0]
artifact = {
    "control": "--no-op-offload",
    "supported_values": [0, 1],
    "default": int(m.group(1)),
    "laguna_configures": "no-op-offload" in env_text or "no-op-offload" in common,
    "laguna_records": "op_offload" in metrics or "no_op_offload" in metrics,
    "semantic_note": "0 leaves operation offload enabled; 1 disables it",
    "binary": bench,
}
out = root / "results/op-offload-policy-audit-20260807.json"
out.write_text(json.dumps(artifact, indent=2) + "\n")
assert artifact["default"] == 0
assert not artifact["laguna_configures"]
assert not artifact["laguna_records"]
print(out)
