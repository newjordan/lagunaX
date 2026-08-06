#!/usr/bin/env python3
"""Audit split-lock/bus-lock detection policy and Laguna benchmark provenance."""
import json
import pathlib
import re
from datetime import datetime, timezone

ROOT = pathlib.Path(__file__).resolve().parents[1]
ENV = (ROOT / "env.sh").read_text()
BENCH = (ROOT / "scripts/bench-serial.sh").read_text()
CMDLINE = pathlib.Path("/proc/cmdline").read_text().strip()
DMESG = pathlib.Path("/proc/sys/kernel/dmesg_restrict").read_text().strip()
SOURCE = ENV + "\n" + BENCH

keys = ("split_lock_detect", "bus_lock_detect")
boot = {}
for key in keys:
    match = re.search(rf"(?:^|\s){key}=([^\s]+)", CMDLINE)
    boot[key] = match.group(1) if match else None

payload = {
    "generated_utc": datetime.now(timezone.utc).isoformat(),
    "angle": "kernel-split-bus-lock-detection-policy",
    "live_state": {
        "kernel_cmdline": CMDLINE,
        "boot_overrides": boot,
        "dmesg_restrict": int(DMESG),
        "effective_policy": {
            "split_lock_detect": boot["split_lock_detect"] or "kernel-default",
            "bus_lock_detect": boot["bus_lock_detect"] or "kernel-default",
        },
    },
    "canonical_policy": {
        "env_or_harness_controls_lock_detection": any(re.search(rf"\b{key}\b", SOURCE) for key in keys),
        "metrics_record_lock_detection": any(key in BENCH for key in ("split_lock_detect", "bus_lock_detect", "split_lock_events", "bus_lock_events")),
        "rejects_lock_events": any(key in BENCH for key in ("split lock", "bus lock", "split_lock", "bus_lock")),
    },
}
out = ROOT / "results/kernel-split-bus-lock-detection-policy-audit-20260807.json"
out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
print(out)
assert all(value is None for value in boot.values())
assert not any(payload["canonical_policy"].values())
