#!/usr/bin/env python3
"""Audit model loading / direct-I/O policy for the active serial Laguna benchmark."""
import json
import os
import re
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
env_text = (ROOT / "env.sh").read_text()
bench_text = (ROOT / "scripts/bench-serial.sh").read_text()
binary = os.environ.get("LX_LLAMA_BENCH")
if not binary:
    raise SystemExit("source env.sh before running this audit")
help_text = subprocess.run([binary, "--help"], text=True, stdout=subprocess.PIPE,
                           stderr=subprocess.STDOUT, check=True).stdout
match = re.search(r"--load-mode <([^>]+)>\s+\(default: ([^)]+)\)", help_text)
if not match:
    raise SystemExit("active llama-bench does not expose the expected --load-mode help")
modes = match.group(1).split("|")
default = match.group(2)
source_override = bool(re.search(r"(?:--load-mode|-lm|--direct-io|-dio|--mmap|-mmp)(?:\s|\")", env_text + "\n" + bench_text))
env_override = any(os.environ.get(k) is not None for k in ("LX_LOAD_MODE", "LOAD_MODE", "DIRECT_IO"))
artifacts = 0
load_mode_records = 0
for path in list((ROOT / "results").rglob("*.json")) + list((ROOT / "baseline").rglob("*.json")):
    try:
        obj = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        continue
    artifacts += 1
    if re.search(r'"(?:load_mode|load-mode|direct_io|mmap)"\s*:', json.dumps(obj)):
        load_mode_records += 1
payload = {
    "binary": binary,
    "supported_load_modes": modes,
    "executable_default": default,
    "active_source_override": source_override,
    "active_environment_override": env_override,
    "effective_policy": default if not source_override and not env_override else "overridden",
    "parsed_json_artifacts": artifacts,
    "artifacts_recording_load_policy": load_mode_records,
}
out = ROOT / "results/model-load-io-policy-audit-20260807.json"
out.write_text(json.dumps(payload, indent=2) + "\n")
print(json.dumps(payload, indent=2))
