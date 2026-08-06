#!/usr/bin/env python3
"""Audit kernel timer/tick policy and Laguna benchmark provenance."""
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
HARNESS = ROOT / "scripts" / "bench-serial.sh"
ENV = ROOT / "env.sh"
OUT = ROOT / "results" / "kernel-timer-tick-policy-audit-20260807.json"


def read(path: str):
    p = Path(path)
    return p.read_text().strip() if p.exists() else None


cmdline = read("/proc/cmdline") or ""
clocksource = read("/sys/devices/system/clocksource/clocksource0/current_clocksource")
available_clocksources = (read("/sys/devices/system/clocksource/clocksource0/available_clocksource") or "").split()
nohz_full = read("/sys/devices/system/cpu/nohz_full")
isolated = read("/sys/devices/system/cpu/isolated")
source = HARNESS.read_text() + "\n" + ENV.read_text()
controls = ("nohz_full", "isolcpus", "rcu_nocbs", "clocksource", "tsc", "hpet")
configured = {name: bool(re.search(rf"(?<![A-Za-z0-9_]){re.escape(name)}(?![A-Za-z0-9_])", source, re.I)) for name in controls}

artifact = {
    "schema": "laguna.kernel-timer-tick-policy-audit.v1",
    "kernel_cmdline": cmdline,
    "clocksource": {
        "current": clocksource,
        "available": available_clocksources,
    },
    "tick_isolation": {
        "nohz_full_sysfs": nohz_full,
        "isolated_cpus_sysfs": isolated,
        "cmdline_nohz_full": bool(re.search(r"(?:^|\s)nohz_full=", cmdline)),
        "cmdline_isolcpus": bool(re.search(r"(?:^|\s)isolcpus=", cmdline)),
        "cmdline_rcu_nocbs": bool(re.search(r"(?:^|\s)rcu_nocbs=", cmdline)),
    },
    "laguna": {
        "controls_present": configured,
        "explicit_policy_configured": any(configured.values()),
        "metrics_record_effective_policy": bool(re.search(r"nohz|isolcpus|rcu_nocbs|clocksource", HARNESS.read_text(), re.I)),
    },
}
assert clocksource
assert available_clocksources
assert clocksource in available_clocksources
assert not artifact["laguna"]["explicit_policy_configured"]
assert not artifact["laguna"]["metrics_record_effective_policy"]
OUT.parent.mkdir(parents=True, exist_ok=True)
OUT.write_text(json.dumps(artifact, indent=2) + "\n")
print(OUT)
