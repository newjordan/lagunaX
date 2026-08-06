#!/usr/bin/env python3
"""Audit active llama-bench host polling and scheduling-priority policy."""
import json
import os
import re
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ENV = (ROOT / "env.sh").read_text()
BENCH = (ROOT / "scripts/bench-serial.sh").read_text()
exe = os.environ.get("LX_LLAMA_BENCH")
if not exe:
    match = re.search(r'export LX_LLAMA_BENCH="([^"]+)"', ENV)
    if not match:
        raise SystemExit("LX_LLAMA_BENCH is unavailable")
    exe = os.path.expandvars(match.group(1))
help_text = subprocess.run([exe, "--help"], text=True, stdout=subprocess.PIPE,
                           stderr=subprocess.STDOUT, check=True).stdout

def default(pattern: str, name: str) -> int:
    match = re.search(pattern, help_text)
    if not match:
        raise SystemExit(f"unable to derive {name} default")
    return int(match.group(1))

poll_default = default(r"--poll <0\.\.\.100>.*?\(default: (-?\d+)\)", "poll")
priority_default = default(r"--prio <-1\|0\|1\|2\|3>.*?\(default: (-?\d+)\)", "priority")
active = ENV + "\n" + BENCH
result = {
    "executable": exe,
    "supported_controls": {"poll": "--poll", "priority": "--prio"},
    "defaults": {"poll_percent": poll_default, "priority": priority_default},
    "active_overrides": {
        "poll": bool(re.search(r"(^|\s)--poll(?:\s|=)", active)),
        "priority": bool(re.search(r"(^|\s)--prio(?:\s|=)", active)),
    },
    "effective": {"poll_percent": poll_default, "priority": priority_default},
}
out = ROOT / "results/host-poll-priority-policy-audit-20260807.json"
out.write_text(json.dumps(result, indent=2) + "\n")
print(out)
