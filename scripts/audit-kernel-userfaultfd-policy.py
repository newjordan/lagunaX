#!/usr/bin/env python3
"""Audit userfaultfd policy and Laguna benchmark provenance."""
import json
import re
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ENV = ROOT / "env.sh"
HARNESS = ROOT / "scripts/bench-serial.sh"
OUT = ROOT / "results/kernel-userfaultfd-policy-audit-20260807.json"


def read(path: str) -> str | None:
    try:
        return Path(path).read_text().strip()
    except (FileNotFoundError, PermissionError):
        return None


raw = read("/proc/sys/vm/unprivileged_userfaultfd")
if raw is None:
    raise RuntimeError("kernel does not export vm.unprivileged_userfaultfd")
value = int(raw)
env_text = ENV.read_text(errors="replace")
harness_text = HARNESS.read_text(errors="replace")
source = env_text + "\n" + harness_text
control_pattern = re.compile(r"unprivileged_userfaultfd|vm\.unprivileged_userfaultfd|userfaultfd", re.I)
metric_pattern = re.compile(r"unprivileged_userfaultfd|userfaultfd_(?:policy|state|events)", re.I)

report = {
    "angle": "kernel-userfaultfd-policy",
    "generated_at": datetime.now(timezone.utc).isoformat(),
    "live": {
        "unprivileged_userfaultfd": value,
        "unprivileged_userfaultfd_enabled": value == 1,
        "proc_interface": "/proc/sys/vm/unprivileged_userfaultfd",
    },
    "laguna": {
        "env_or_harness_controls_userfaultfd": bool(control_pattern.search(source)),
        "serial_harness_records_userfaultfd_policy": bool(metric_pattern.search(harness_text)),
    },
    "interpretation": (
        "userfaultfd lets user space handle page faults. This audit records whether unprivileged "
        "processes may use it and whether Laguna controls or records that policy."
    ),
}
assert value in (0, 1)
assert not report["laguna"]["env_or_harness_controls_userfaultfd"]
assert not report["laguna"]["serial_harness_records_userfaultfd_policy"]
OUT.parent.mkdir(parents=True, exist_ok=True)
OUT.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
print(OUT)
