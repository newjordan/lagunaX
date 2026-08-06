#!/usr/bin/env python3
"""Audit kernel lockup-watchdog policy and Laguna benchmark provenance."""
import json
import re
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ENV = ROOT / "env.sh"
HARNESS = ROOT / "scripts/bench-serial.sh"
OUT = ROOT / "results/kernel-lockup-watchdog-policy-audit-20260807.json"


def read(path: str) -> str | None:
    try:
        return Path(path).read_text().strip()
    except (FileNotFoundError, PermissionError):
        return None


interfaces = {
    "watchdog_enabled": "/proc/sys/kernel/watchdog",
    "nmi_watchdog_enabled": "/proc/sys/kernel/nmi_watchdog",
    "watchdog_threshold_seconds": "/proc/sys/kernel/watchdog_thresh",
    "soft_watchdog_enabled": "/proc/sys/kernel/soft_watchdog",
    "hardlockup_panic_enabled": "/proc/sys/kernel/hardlockup_panic",
    "softlockup_panic_enabled": "/proc/sys/kernel/softlockup_panic",
}
live = {name: (int(value) if (value := read(path)) is not None else None)
        for name, path in interfaces.items()}
cmdline = read("/proc/cmdline") or ""
env_text = ENV.read_text(errors="replace")
harness_text = HARNESS.read_text(errors="replace")
source = env_text + "\n" + harness_text
control_pattern = re.compile(r"(?:nmi_)?watchdog|watchdog_thresh|softlockup|hardlockup", re.I)
metric_pattern = re.compile(r"/proc/sys/kernel/(?:nmi_)?watchdog|watchdog_thresh|soft_watchdog|lockup", re.I)
boot_overrides = re.findall(r"(?:^|\s)(?:nmi_watchdog|nowatchdog|softlockup_panic|hardlockup_panic)=[^\s]+", cmdline)

report = {
    "angle": "kernel-lockup-watchdog-policy",
    "generated_at": datetime.now(timezone.utc).isoformat(),
    "live": live,
    "kernel_command_line_watchdog_overrides": boot_overrides,
    "interfaces": interfaces,
    "laguna": {
        "env_or_harness_controls_watchdogs": bool(control_pattern.search(source)),
        "serial_harness_records_watchdog_policy": bool(metric_pattern.search(harness_text)),
    },
    "interpretation": (
        "Kernel lockup watchdogs periodically inspect CPU progress and may use performance-monitoring "
        "interrupts. This audit records their live policy and Laguna's control/provenance coverage."
    ),
}
assert live["watchdog_enabled"] in (0, 1)
assert live["nmi_watchdog_enabled"] in (0, 1)
assert live["watchdog_threshold_seconds"] is not None
assert not report["laguna"]["env_or_harness_controls_watchdogs"]
assert not report["laguna"]["serial_harness_records_watchdog_policy"]
OUT.parent.mkdir(parents=True, exist_ok=True)
OUT.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
print(OUT)
