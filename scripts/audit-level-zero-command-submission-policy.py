#!/usr/bin/env python3
"""Audit Level Zero immediate-command-list policy and benchmark provenance."""
import json
import os
import pathlib
import re
import subprocess

root = pathlib.Path(__file__).resolve().parents[1]
env_text = (root / "env.sh").read_text()
harness_text = (root / "scripts/bench-serial.sh").read_text()
keys = (
    "SYCL_PI_LEVEL_ZERO_USE_IMMEDIATE_COMMANDLISTS",
    "UR_L0_USE_IMMEDIATE_COMMANDLISTS",
    "SYCL_PI_LEVEL_ZERO_DEVICE_SCOPE_EVENTS",
    "UR_L0_USE_DRIVER_INORDER_LISTS",
)

raw = subprocess.check_output(
    ["bash", "-c", "source ./env.sh >/dev/null 2>&1; env -0"], cwd=root
)
active = dict(
    item.split("=", 1)
    for item in raw.decode(errors="replace").split("\0")
    if "=" in item
)
report = {
    "angle": "level-zero-command-submission-policy",
    "active_policy": {key: active.get(key) for key in keys},
    "source_mentions": {
        key: {
            "env_sh": bool(re.search(rf"\b{re.escape(key)}\b", env_text)),
            "serial_harness": bool(re.search(rf"\b{re.escape(key)}\b", harness_text)),
        }
        for key in keys
    },
    "provenance": {
        "serial_harness_records_any_command_submission_control": any(
            key in harness_text for key in keys
        )
    },
}
out = root / "benchmark/results/level-zero-command-submission-policy-audit-20260807.json"
out.parent.mkdir(parents=True, exist_ok=True)
out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
print(out)
