#!/usr/bin/env python3
"""Audit kernel preemption model, timer frequency, and Laguna provenance."""
import json
import re
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ENV = (ROOT / "env.sh").read_text(errors="replace")
BENCH = (ROOT / "scripts/bench-serial.sh").read_text(errors="replace")


def read(path):
    try:
        return path.read_text(errors="replace").strip()
    except OSError:
        return None


release = read(Path("/proc/sys/kernel/osrelease"))
version = read(Path("/proc/version"))
cmdline = read(Path("/proc/cmdline")) or ""
config_path = Path(f"/boot/config-{release}")
config_text = read(config_path) or ""
keys = (
    "CONFIG_PREEMPT_BUILD",
    "CONFIG_PREEMPT",
    "CONFIG_PREEMPT_LAZY",
    "CONFIG_PREEMPT_RT",
    "CONFIG_PREEMPT_DYNAMIC",
    "CONFIG_HZ_100",
    "CONFIG_HZ_250",
    "CONFIG_HZ_300",
    "CONFIG_HZ_1000",
)
config = {}
for key in keys:
    match = re.search(rf"^(?:{re.escape(key)}=(.+)|# {re.escape(key)} is not set)$", config_text, re.M)
    config[key] = match.group(1) if match and match.group(1) is not None else ("n" if match else None)

source = ENV + "\n" + BENCH
control_terms = ("preempt=", "PREEMPT_DYNAMIC", "PREEMPT_LAZY", "CONFIG_HZ")
metric_terms = ("kernel_preemption_model", "kernel_preempt_dynamic", "kernel_timer_hz")
report = {
    "generated_utc": datetime.now(timezone.utc).isoformat(),
    "angle": "kernel-preemption-model-policy",
    "live_policy": {
        "kernel_release": release,
        "kernel_version": version,
        "boot_config_path": str(config_path),
        "boot_config": config,
        "kernel_cmdline_preempt_controls": [word for word in cmdline.split() if word.startswith("preempt=")],
        "debugfs_runtime_preempt": read(Path("/sys/kernel/debug/sched/preempt")),
    },
    "canonical_policy": {
        "env_or_harness_controls_preemption": [term for term in control_terms if term in source],
        "metrics_record_preemption": [term for term in metric_terms if f'"{term}"' in BENCH],
    },
}

assert release
assert version
assert config_text, f"kernel config unreadable: {config_path}"
assert config["CONFIG_PREEMPT_DYNAMIC"] == "y"
assert config["CONFIG_HZ_1000"] == "y"
assert not report["canonical_policy"]["env_or_harness_controls_preemption"]
assert not report["canonical_policy"]["metrics_record_preemption"]
out = ROOT / "results/kernel-preemption-model-policy-audit-20260807.json"
out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
print(out)
