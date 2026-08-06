#!/usr/bin/env python3
"""Audit whether Laguna forbids host-buffer fallback in llama-bench."""
import json
import os
import re
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
env_text = (ROOT / "env.sh").read_text()
bench_text = (ROOT / "scripts/bench-serial.sh").read_text()
bench = os.environ.get("LX_LLAMA_BENCH")
if not bench:
    m = re.search(r'export LX_BIN="\$\{LX_BIN:-([^}]+)\}"', env_text)
    if not m:
        raise SystemExit("cannot resolve LX_BIN")
    bench = str(Path(m.group(1)) / "llama-bench")
help_text = subprocess.run([bench, "--help"], text=True, capture_output=True, check=True).stdout
match = re.search(r"--no-host <0\|1>\s+\(default: ([01])\)", help_text)
if not match:
    raise SystemExit("--no-host policy absent from executable help")
source_override = bool(re.search(r"(?:--no-host)(?:\s|=)", env_text + "\n" + bench_text))
artifact_mentions = 0
parsed = 0
for path in ROOT.rglob("*.json"):
    try:
        text = path.read_text()
        json.loads(text)
        parsed += 1
        artifact_mentions += int("no-host" in text or "no_host" in text)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        pass
payload = {
    "executable": bench,
    "executable_default_no_host": bool(int(match.group(1))),
    "source_override": source_override,
    "effective_no_host": bool(int(match.group(1))) if not source_override else None,
    "meaning": "host-buffer fallback remains permitted" if match.group(1) == "0" and not source_override else "explicit host-buffer restriction requires inspection",
    "parsed_json_artifacts": parsed,
    "artifacts_mentioning_control": artifact_mentions,
}
out = ROOT / "results" / "no-host-buffer-policy-audit-20260807.json"
out.write_text(json.dumps(payload, indent=2) + "\n")
print(out)
print(json.dumps(payload, indent=2))
