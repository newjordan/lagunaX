#!/usr/bin/env python3
"""Audit model loading and NUMA policy for the active Laguna benchmark."""
import json
import re
import subprocess
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
env_text = (ROOT / "env.sh").read_text()
bench_text = (ROOT / "scripts/bench-serial.sh").read_text()
run = subprocess.run(
    ["bash", "-c", 'source "$1" >/dev/null 2>&1; "$LX_LLAMA_BENCH" --help', "audit", str(ROOT / "env.sh")],
    text=True, capture_output=True,
)
help_text = run.stdout + run.stderr
load_line = next((x.strip() for x in help_text.splitlines() if "--load-mode" in x), "")
numa_line = next((x.strip() for x in help_text.splitlines() if "--numa" in x), "")
if not load_line or not numa_line:
    raise SystemExit("active llama-bench does not advertise load-mode and NUMA controls")

def controlled(pattern: str, text: str) -> bool:
    return bool(re.search(pattern, text))

load_pattern = r"(?:--load-mode|-lm|--mmap|-mmp|--direct-io|-dio)(?:\s|=)"
numa_pattern = r"--numa(?:\s|=)"
artifacts = mentions = 0
for path in (ROOT / "results").rglob("*.json"):
    try:
        serialized = json.dumps(json.loads(path.read_text())).lower()
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        continue
    artifacts += 1
    if any(x in serialized for x in ("load_mode", "load-mode", "mmap", "mlock", "direct_io", "direct-io", "numa")):
        mentions += 1
payload = {
    "timestamp_utc": datetime.now(timezone.utc).isoformat(),
    "load_mode_help": load_line,
    "numa_help": numa_line,
    "default_load_mode": "mmap",
    "default_numa_mode": "disabled",
    "env_load_override": controlled(load_pattern, env_text),
    "harness_load_override": controlled(load_pattern, bench_text),
    "env_numa_override": controlled(numa_pattern, env_text),
    "harness_numa_override": controlled(numa_pattern, bench_text),
    "historical_json_artifacts_parsed": artifacts,
    "historical_policy_mentions": mentions,
}
out = ROOT / "results/model-load-numa-policy-audit-20260807.json"
out.write_text(json.dumps(payload, indent=2) + "\n")
print(out)
