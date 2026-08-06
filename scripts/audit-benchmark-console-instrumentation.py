#!/usr/bin/env python3
"""Audit benchmark console/progress instrumentation that can perturb timing."""
import json
import os
import pathlib
import re
import subprocess

root = pathlib.Path(__file__).resolve().parents[1]
env_text = (root / "env.sh").read_text()
harness_text = (root / "scripts/bench-serial.sh").read_text()
m = re.search(r'^export LX_LLAMA_BENCH="\$\{LX_LLAMA_BENCH:-([^}]+)\}"', env_text, re.M)
if not m:
    raise SystemExit("LX_LLAMA_BENCH not found")
bin_path = m.group(1).replace("$LX_BIN", re.search(r'^export LX_BIN="\$\{LX_BIN:-([^}]+)\}"', env_text, re.M).group(1))
help_text = subprocess.run(
    ["bash", "-lc", f"source {root / 'env.sh'} && \"$LX_LLAMA_BENCH\" --help 2>&1 || true"],
    text=True, stdout=subprocess.PIPE).stdout
common = re.search(r'COMMON=\((.*?)\n\s*\)', harness_text, re.S).group(1)
result = {
    "executable": bin_path,
    "capabilities": {
        "verbose": "--verbose" in help_text,
        "progress": "--progress" in help_text,
        "stderr_output": "--output-err" in help_text,
    },
    "active_policy": {
        "verbose_enabled": bool(re.search(r'(^|\s)(-v|--verbose)(\s|$)', common)),
        "progress_enabled": "--progress" in common,
        "stderr_output_configured": bool(re.search(r'(^|\s)(-oe|--output-err)(\s|$)', common)),
        "stdout_output": re.search(r'-o\s+(\S+)', common).group(1),
        "stderr_is_captured_in_timed_invocations": '2>>"$RAW_LOG"' in harness_text,
    },
    "defaults": {"verbose": False, "progress": False, "stderr_output": "none"},
}
out = root / "benchmark/results/benchmark-console-instrumentation-audit-20260807.json"
out.parent.mkdir(parents=True, exist_ok=True)
out.write_text(json.dumps(result, indent=2) + "\n")
print(out)
