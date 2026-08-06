#!/usr/bin/env python3
"""Audit accelerator split and primary-device policy and provenance."""
import json
import os
import re
import subprocess
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ENV = (ROOT / "env.sh").read_text()
BENCH = (ROOT / "scripts/bench-serial.sh").read_text()

def env_value(name):
    cmd = f'source "{ROOT}/env.sh" >/dev/null 2>&1; printf %s "${{{name}}}"'
    return subprocess.check_output(["bash", "-c", cmd], text=True)

binary = env_value("LX_LLAMA_BENCH")
help_text = subprocess.run(
    ["bash", "-c", f'source "{ROOT}/env.sh" >/dev/null 2>&1; exec "$LX_LLAMA_BENCH" --help'],
    text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, check=False,
).stdout
patterns = {
    "split_mode": r"--split-mode <([^>]+)>\s+\(default: ([^)]+)\)",
    "main_gpu": r"--main-gpu <i>\s+\(default: ([^)]+)\)",
    "tensor_split": r"--tensor-split <[^>]+>\s+\(default: ([^)]+)\)",
}
parsed = {}
for key, pattern in patterns.items():
    match = re.search(pattern, help_text)
    if not match:
        raise AssertionError(f"missing live help contract for {key}")
    parsed[key] = {"choices": match.group(1), "default": match.group(2)} if key == "split_mode" else {"default": match.group(1)}

needles = ("--split-mode", "-sm", "--main-gpu", "-mg", "--tensor-split", "-ts")
payload = {
    "generated_utc": datetime.now(timezone.utc).isoformat(),
    "binary": binary,
    "live_contract": parsed,
    "canonical_policy": {
        "env_configures_split_or_primary_device": any(x in ENV for x in ("SPLIT_MODE", "MAIN_GPU", "TENSOR_SPLIT")),
        "harness_passes_split_or_primary_device": any(x in BENCH for x in needles),
        "metrics_record_split_or_primary_device": any(x in BENCH for x in ("split_mode", "main_gpu", "tensor_split")),
    },
    "active_device_selection": {
        "ONEAPI_DEVICE_SELECTOR": os.environ.get("ONEAPI_DEVICE_SELECTOR", env_value("ONEAPI_DEVICE_SELECTOR")),
        "ZE_AFFINITY_MASK": os.environ.get("ZE_AFFINITY_MASK", env_value("ZE_AFFINITY_MASK")),
    },
}
out = ROOT / "results/accelerator-split-primary-device-policy-audit-20260807.json"
out.write_text(json.dumps(payload, indent=2) + "\n")
print(out)
assert parsed["split_mode"]["default"] == "layer"
assert parsed["main_gpu"]["default"] == "0"
assert parsed["tensor_split"]["default"] == "0"
assert not any(payload["canonical_policy"].values())
