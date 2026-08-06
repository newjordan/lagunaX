#!/usr/bin/env python3
"""Audit host worker polling and scheduling-priority policy for Laguna."""
import json
import os
import pathlib
import re
import subprocess

root = pathlib.Path(__file__).resolve().parents[1]
env_text = (root / "env.sh").read_text()
bench_text = (root / "scripts/bench-serial.sh").read_text()
env = os.environ.copy()
env.setdefault("LX_ROOT", str(root))
# Resolve the executable from the active shell policy without executing benchmarks.
resolved = subprocess.check_output(
    ["bash", "-c", f"source {root / 'env.sh'} >/dev/null 2>&1; printf %s \"$LX_LLAMA_BENCH\""],
    text=True,
)
help_run = subprocess.run(
    ["bash", "-c", f"source {root / 'env.sh'} >/dev/null 2>&1; exec \"$LX_LLAMA_BENCH\" --help"],
    text=True,
    stdout=subprocess.PIPE,
    stderr=subprocess.STDOUT,
    check=False,
)
help_text = help_run.stdout
if not help_text:
    raise RuntimeError(f"llama-bench help produced no output (exit {help_run.returncode})")

def default(pattern: str):
    match = re.search(pattern, help_text, re.MULTILINE)
    return int(match.group(1)) if match else None

active = env_text + "\n" + bench_text
artifact = {
    "policy": "host-worker-wait-and-priority",
    "executable": resolved,
    "executable_defaults": {
        "poll_percent": default(r"--poll <0\.\.\.100>.*?\(default: (-?\d+)\)"),
        "priority": default(r"--prio <-1\|0\|1\|2\|3>.*?\(default: (-?\d+)\)"),
    },
    "active_overrides": {
        "poll": bool(re.search(r"(^|\s)--poll(?:\s|=)", active)),
        "priority": bool(re.search(r"(^|\s)--prio(?:\s|=)", active)),
    },
}
assert artifact["executable_defaults"] == {"poll_percent": 50, "priority": 0}
assert artifact["active_overrides"] == {"poll": False, "priority": False}
out = root / "results" / "host-worker-wait-priority-audit-20260807.json"
out.write_text(json.dumps(artifact, indent=2) + "\n")
print(out)
