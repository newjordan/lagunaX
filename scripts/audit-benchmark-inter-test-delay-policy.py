#!/usr/bin/env python3
"""Audit inter-test delay policy in the canonical Laguna serial benchmark."""
import datetime as dt
import hashlib
import json
import os
import pathlib
import re
import subprocess

ROOT = pathlib.Path(__file__).resolve().parents[1]
env_text = (ROOT / "env.sh").read_text()
resolved_env = subprocess.run(
    ["bash", "-c", 'source "$1" >/dev/null 2>&1; printf "%s" "$LX_LLAMA_BENCH"', "bash", str(ROOT / "env.sh")],
    text=True, capture_output=True, check=True,
).stdout
bench_path = pathlib.Path(resolved_env)
serial_path = ROOT / "scripts/bench-serial.sh"
serial_text = serial_path.read_text()
if not bench_path or not bench_path.is_file():
    raise SystemExit(f"active executable not found: {bench_path}")
help_run = subprocess.run(
    ["bash", "-c", 'source "$1" >/dev/null 2>&1; "$LX_LLAMA_BENCH" --help', "bash", str(ROOT / "env.sh")],
    text=True, capture_output=True,
)
help_text = help_run.stdout + help_run.stderr
match = re.search(r'--delay <0\.\.\.N> \(seconds\)\s+delay between each test \(default: (\d+)\)', help_text)
if not match:
    raise SystemExit("unable to derive --delay default")
source_override = bool(re.search(r'(^|[\s"\'])--delay(?:[\s="\']|$)', serial_text))
env_override = "LLAMA_ARG_DELAY" in os.environ or "LX_DELAY" in os.environ
artifact = {
    "audit": "benchmark-inter-test-delay-policy",
    "timestamp_utc": dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
    "executable": str(bench_path),
    "executable_sha256": hashlib.sha256(bench_path.read_bytes()).hexdigest(),
    "delay_supported": True,
    "executable_default_delay_seconds": int(match.group(1)),
    "serial_source_override": source_override,
    "process_environment_override": env_override,
    "effective_delay_seconds": None if source_override or env_override else int(match.group(1)),
    "separate_processes_mean_delay_is_not_applied_between_pp_and_tg": True,
}
out = ROOT / "results" / f"benchmark-inter-test-delay-policy-audit-{dt.datetime.now(dt.timezone.utc):%Y%m%d}.json"
out.write_text(json.dumps(artifact, indent=2) + "\n")
print(out.relative_to(ROOT))
print(json.dumps(artifact, indent=2))
