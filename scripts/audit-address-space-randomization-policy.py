#!/usr/bin/env python3
"""Audit process/kernel address-space randomization policy and Laguna provenance."""
import json
import pathlib

ROOT = pathlib.Path(__file__).resolve().parents[1]


def read_optional(path: pathlib.Path):
    try:
        return path.read_text().strip()
    except (FileNotFoundError, PermissionError, OSError):
        return None


sysctl = read_optional(pathlib.Path("/proc/sys/kernel/randomize_va_space"))
status = read_optional(pathlib.Path("/proc/self/personality"))
if status is None:
    # /proc/self/personality is not available on every kernel; maps still prove
    # whether this process received ordinary randomized mappings.
    status = "unavailable"
maps = read_optional(pathlib.Path("/proc/self/maps")) or ""
map_rows = [line for line in maps.splitlines() if line]
env_text = (ROOT / "env.sh").read_text()
harness_text = (ROOT / "scripts/bench-serial.sh").read_text()
terms = ("randomize_va_space", "setarch", "ADDR_NO_RANDOMIZE", "personality(", "--addr-no-randomize")
report = {
    "angle": "address-space-layout-randomization-policy-and-provenance",
    "kernel_randomize_va_space": int(sysctl) if sysctl is not None else None,
    "kernel_policy_meaning": {
        0: "disabled",
        1: "conservative randomization",
        2: "full randomization including heap",
    }.get(int(sysctl) if sysctl is not None else None, "unavailable-or-unknown"),
    "proc_self_personality": status,
    "process_mapping_count": len(map_rows),
    "has_vdso_mapping": any("[vdso]" in row for row in map_rows),
    "provenance": {
        "env_controls_aslr": any(term in env_text for term in terms),
        "serial_harness_controls_or_records_aslr": any(term in harness_text for term in terms),
    },
}
out = ROOT / "benchmark/results/address-space-randomization-policy-audit-20260807.json"
out.parent.mkdir(parents=True, exist_ok=True)
out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
print(out)
