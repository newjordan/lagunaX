#!/usr/bin/env python3
"""Audit Linux VM allocator/reclaim tuning and Laguna provenance."""
import json
import pathlib

ROOT = pathlib.Path(__file__).resolve().parents[1]
VM = pathlib.Path("/proc/sys/vm")
KNOBS = (
    "zone_reclaim_mode",
    "watermark_scale_factor",
    "watermark_boost_factor",
    "stat_interval",
    "percpu_pagelist_high_fraction",
    "numa_stat",
)


def read_int(name):
    try:
        return int((VM / name).read_text().strip())
    except (FileNotFoundError, PermissionError, OSError, ValueError):
        return None


env_text = (ROOT / "env.sh").read_text().lower()
harness_text = (ROOT / "scripts/bench-serial.sh").read_text().lower()
report = {
    "angle": "linux-vm-zone-watermark-and-percpu-page-list-policy",
    "live": {name: read_int(name) for name in KNOBS},
    "provenance": {
        "env_controls_policy": any(name in env_text for name in KNOBS),
        "serial_harness_records_policy": any(name in harness_text for name in KNOBS),
    },
}
out = ROOT / "benchmark/results/kernel-vm-zone-watermark-policy-audit-20260807.json"
out.parent.mkdir(parents=True, exist_ok=True)
out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
print(out)
