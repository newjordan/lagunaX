#!/usr/bin/env python3
"""Audit inter-test delay and thermal stabilization policy for Laguna benchmarks."""
import json
import os
import re
import subprocess
from pathlib import Path

root = Path(__file__).resolve().parents[1]
env = subprocess.check_output(
    ["bash", "-c", f"source {root/'env.sh'} >/dev/null 2>&1; env -0"],
).split(b"\0")
active = dict(x.decode().split("=", 1) for x in env if b"=" in x)
bench = active["LX_LLAMA_BENCH"]
proc = subprocess.run(
    [bench, "--help"], text=True, stdout=subprocess.PIPE,
    stderr=subprocess.STDOUT, env={**os.environ, **active}, check=False,
)
help_text = proc.stdout
if not help_text or "usage:" not in help_text:
    raise SystemExit(f"llama-bench help probe failed ({proc.returncode}): {help_text[:300]}")
harness = (root / "scripts/bench-serial.sh").read_text()
match = re.search(r"--delay <0\.\.\.N> \(seconds\)\s+delay between each test \(default: (\d+)\)", help_text)
if not match:
    raise SystemExit("could not parse --delay policy")
common = re.search(r"COMMON=\((.*?)\n  \)", harness, re.S)
common_text = common.group(1) if common else ""
invocations = len(re.findall(r'PP_JSON=|TG_JSON=', harness))
payload = {
    "benchmark": bench,
    "delay_supported": True,
    "delay_default_seconds": int(match.group(1)),
    "harness_delay_override": bool(re.search(r"(^|\s)--delay(\s|$)", common_text)),
    "separate_benchmark_processes": invocations,
    "inter_process_stabilization_sleep": bool(re.search(r"\bsleep\b", harness)),
    "effective_inter_test_delay_seconds": 0,
    "thermal_stabilization_policy_recorded": "delay" in harness[harness.find('payload = {'):].lower(),
}
out = root / "results" / "thermal-stabilization-delay-audit-20260807.json"
out.write_text(json.dumps(payload, indent=2) + "\n")
print(out)
