#!/usr/bin/env python3
"""Audit strict CPU-affinity enforcement independently of the selected mask."""
import json
import re
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
env_text = (ROOT / "env.sh").read_text()
bench_text = (ROOT / "scripts/bench-serial.sh").read_text()

resolved = subprocess.run(
    ["bash", "-c", 'source "$1" >/dev/null 2>&1; printf "%s" "$LX_LLAMA_BENCH"', "bash", str(ROOT / "env.sh")],
    check=True, text=True, stdout=subprocess.PIPE,
).stdout
if not resolved:
    raise SystemExit("LX_LLAMA_BENCH did not resolve after sourcing env.sh")
help_text = subprocess.run(
    ["bash", "-c", 'source "$1" >/dev/null 2>&1; "$LX_LLAMA_BENCH" --help', "bash", str(ROOT / "env.sh")],
    check=True, text=True,
                           stdout=subprocess.PIPE, stderr=subprocess.STDOUT).stdout
option = re.search(r'^\s*--cpu-strict <0\|1>\s+\(default: (\d)\)', help_text, re.M)
if not option:
    raise SystemExit("--cpu-strict help contract not found")
source_override = bool(re.search(r'--cpu-strict\b|CPU_STRICT', env_text + "\n" + bench_text))
metrics_recorded = bool(re.search(r'["\']cpu_strict["\']', bench_text))
result = {
    "audit": "strict-cpu-affinity-enforcement-policy",
    "executable": resolved,
    "supported": True,
    "default": int(option.group(1)),
    "active_source_override": source_override,
    "effective_policy": "advisory/non-strict" if not source_override and option.group(1) == "0" else "overridden",
    "canonical_metrics_record_policy": metrics_recorded,
}
assert result["default"] == 0
assert not result["active_source_override"]
assert not result["canonical_metrics_record_policy"]
out = ROOT / "benchmark/results/strict-cpu-affinity-enforcement-audit-20260807.json"
out.parent.mkdir(parents=True, exist_ok=True)
out.write_text(json.dumps(result, indent=2) + "\n")
print(out)
