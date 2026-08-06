#!/usr/bin/env python3
"""Audit CPU idle-state availability, disablement, and Laguna provenance."""
import json
import pathlib

ROOT = pathlib.Path(__file__).resolve().parents[1]
CPU_ROOT = pathlib.Path("/sys/devices/system/cpu")
CMDLINE = pathlib.Path("/proc/cmdline")


def read_optional(path: pathlib.Path):
    try:
        return path.read_text().strip()
    except (FileNotFoundError, PermissionError, OSError):
        return None


states = []
for state_dir in sorted(CPU_ROOT.glob("cpu[0-9]*/cpuidle/state[0-9]*")):
    states.append(
        {
            "cpu": state_dir.parents[1].name,
            "state": state_dir.name,
            "name": read_optional(state_dir / "name"),
            "description": read_optional(state_dir / "desc"),
            "latency_us": read_optional(state_dir / "latency"),
            "residency_us": read_optional(state_dir / "residency"),
            "disabled": read_optional(state_dir / "disable"),
            "usage": read_optional(state_dir / "usage"),
            "time_us": read_optional(state_dir / "time"),
        }
    )

cmdline = read_optional(CMDLINE) or ""
env_text = (ROOT / "env.sh").read_text()
harness_text = (ROOT / "scripts/bench-serial.sh").read_text()
policy_terms = ("cpuidle", "intel_idle", "processor.max_cstate", "idle=poll")
report = {
    "angle": "cpu-idle-state-policy-and-provenance",
    "driver": read_optional(CPU_ROOT / "cpuidle/current_driver"),
    "governor": read_optional(CPU_ROOT / "cpuidle/current_governor_ro"),
    "available_governors": read_optional(CPU_ROOT / "cpuidle/available_governors"),
    "logical_cpus_with_idle_states": len({row["cpu"] for row in states}),
    "exported_state_rows": len(states),
    "state_names": sorted({row["name"] for row in states if row["name"]}),
    "disabled_state_rows": sum(row["disabled"] == "1" for row in states),
    "states": states,
    "boot_overrides": [term for term in policy_terms if term in cmdline],
    "provenance": {
        "env_controls_idle_policy": any(term in env_text for term in policy_terms),
        "serial_harness_records_idle_policy": any(term in harness_text for term in policy_terms),
    },
}
out = ROOT / "benchmark/results/cpu-idle-state-policy-audit-20260807.json"
out.parent.mkdir(parents=True, exist_ok=True)
out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
print(out)
