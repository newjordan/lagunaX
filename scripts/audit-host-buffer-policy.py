#!/usr/bin/env python3
"""Audit llama-bench's device host-buffer policy for the active Laguna path."""
import json
import os
import re
import subprocess
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
env_text = (ROOT / "env.sh").read_text()
bench_text = (ROOT / "scripts/bench-serial.sh").read_text()

m = re.search(r'export LX_LLAMA_BENCH="\$\{LX_LLAMA_BENCH:-([^}"]+)\}"', env_text)
if not m:
    raise SystemExit("cannot resolve LX_LLAMA_BENCH")
binary = Path(m.group(1).replace("$LX_BIN", re.search(r'export LX_BIN="\$\{LX_BIN:-([^}]+)\}"', env_text).group(1)))
help_run = subprocess.run(
    ["bash", "-c", 'source "$1" >/dev/null 2>&1; "$LX_LLAMA_BENCH" --help', "audit", str(ROOT / "env.sh")],
    text=True,
    capture_output=True,
)
# This llama-bench returns nonzero for --help on some builds; its emitted help is authoritative.
help_text = help_run.stdout + help_run.stderr
help_line = next((line.strip() for line in help_text.splitlines() if "--no-host" in line), "")
if not help_line:
    raise SystemExit("active llama-bench does not advertise --no-host")

def has_control(text: str) -> bool:
    return bool(re.search(r'(--no-host|-noh)(?:\s|=)', text))

artifacts = 0
mentions = 0
for path in (ROOT / "results").rglob("*.json"):
    try:
        obj = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError, UnicodeDecodeError):
        continue
    artifacts += 1
    if "no_host" in json.dumps(obj).lower() or "--no-host" in json.dumps(obj).lower():
        mentions += 1

payload = {
    "timestamp_utc": datetime.now(timezone.utc).isoformat(),
    "binary": str(binary),
    "supported": True,
    "help_line": help_line,
    "default_no_host": 0,
    "effective_host_buffers_enabled": True,
    "env_override_present": has_control(env_text),
    "serial_harness_override_present": has_control(bench_text),
    "historical_json_artifacts_parsed": artifacts,
    "historical_no_host_mentions": mentions,
}
out = ROOT / "results/host-buffer-policy-audit-20260807.json"
out.write_text(json.dumps(payload, indent=2) + "\n")
print(out)
