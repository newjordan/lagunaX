#!/usr/bin/env python3
"""Audit memory-locking limits and Laguna benchmark provenance without changing state."""
import json
import pathlib
import resource

ROOT = pathlib.Path(__file__).resolve().parents[1]


def read_text(path):
    try:
        return pathlib.Path(path).read_text().strip()
    except (FileNotFoundError, PermissionError, OSError):
        return None


def limit_value(value):
    return "unlimited" if value == resource.RLIM_INFINITY else value


soft, hard = resource.getrlimit(resource.RLIMIT_MEMLOCK)
status = read_text("/proc/self/status") or ""
vm_lck = next((line.split(":", 1)[1].strip() for line in status.splitlines()
               if line.startswith("VmLck:")), None)
env_text = (ROOT / "env.sh").read_text().lower()
harness_text = (ROOT / "scripts/bench-serial.sh").read_text().lower()
terms = ("memlock", "mlock", "vmlck", "rlimit_memlock")
report = {
    "angle": "process-memory-locking-limit-and-residency-policy",
    "live": {
        "rlimit_memlock_soft_bytes": limit_value(soft),
        "rlimit_memlock_hard_bytes": limit_value(hard),
        "auditor_vm_lck": vm_lck,
    },
    "provenance": {
        "env_controls_or_records_memlock": any(term in env_text for term in terms),
        "serial_harness_controls_or_records_memlock": any(term in harness_text for term in terms),
    },
}
out = ROOT / "benchmark/results/process-memory-locking-policy-audit-20260807.json"
out.parent.mkdir(parents=True, exist_ok=True)
out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
print(out)
