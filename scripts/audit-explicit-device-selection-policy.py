#!/usr/bin/env python3
"""Audit explicit llama-bench device-selection policy and provenance."""
import hashlib
import json
import os
import pathlib
import re
import subprocess

ROOT = pathlib.Path(__file__).resolve().parents[1]
env_text = (ROOT / "env.sh").read_text()
harness_text = (ROOT / "scripts/bench-serial.sh").read_text()
env = os.environ.copy()
probe = subprocess.run(
    ["bash", "-lc", "source ./env.sh; printf '%s\\n' \"$LX_LLAMA_BENCH\" \"$ONEAPI_DEVICE_SELECTOR\" \"$ZE_AFFINITY_MASK\""],
    cwd=ROOT, text=True, capture_output=True, check=True, env=env,
).stdout.splitlines()
bench, selector, affinity = probe
help_run = subprocess.run(
    ["bash", "-lc", "source ./env.sh; \"$LX_LLAMA_BENCH\" --help"],
    cwd=ROOT, text=True, capture_output=True, check=True, env=env,
)
help_text = help_run.stdout + help_run.stderr
m = re.search(r"-dev, --device <dev0/dev1/\.\.\.>\s+\(default: ([^)]+)\)", help_text)
if not m:
    raise SystemExit("could not parse --device default")
listing = subprocess.run(
    ["bash", "-lc", "source ./env.sh; \"$LX_LLAMA_BENCH\" --list-devices"],
    cwd=ROOT, text=True, capture_output=True, check=True, env=env,
)
combined = listing.stdout + listing.stderr
result = {
    "benchmark": bench,
    "benchmark_sha256": hashlib.sha256(pathlib.Path(bench).read_bytes()).hexdigest(),
    "device_option_supported": True,
    "device_option_default": m.group(1),
    "harness_passes_device": bool(re.search(r"(?:^|\s)(?:-dev|--device)(?:\s|$)", harness_text)),
    "harness_records_backend_selector": "ONEAPI_DEVICE_SELECTOR" in harness_text,
    "harness_records_affinity_mask": "ZE_AFFINITY_MASK" in harness_text,
    "oneapi_device_selector": selector,
    "ze_affinity_mask": affinity,
    "env_defines_backend_selector": "ONEAPI_DEVICE_SELECTOR" in env_text,
    "env_defines_affinity_mask": "ZE_AFFINITY_MASK" in env_text,
    "listed_devices_raw": combined.strip().splitlines(),
}
out = ROOT / "results" / "explicit-device-selection-audit-20260807.json"
out.write_text(json.dumps(result, indent=2) + "\n")
print(out)
