#!/usr/bin/env python3
"""Audit process privilege-restriction and seccomp policy without changing state."""
import json
import pathlib

ROOT = pathlib.Path(__file__).resolve().parents[1]
OUT = ROOT / "benchmark/results/process-seccomp-privilege-policy-audit-20260807.json"

status = pathlib.Path("/proc/self/status").read_text()
fields = {}
for line in status.splitlines():
    if ":" in line:
        key, value = line.split(":", 1)
        if key in {"NoNewPrivs", "Seccomp", "Seccomp_filters"}:
            fields[key] = int(value.strip())

env = (ROOT / "env.sh").read_text().lower()
harness = (ROOT / "scripts/bench-serial.sh").read_text().lower()
terms = ("seccomp", "no_new_privs", "nonewprivs", "pr_set_no_new_privs")
report = {
    "angle": "process-seccomp-and-no-new-privileges-policy",
    "live": {
        "no_new_privileges": fields["NoNewPrivs"],
        "seccomp_mode": fields["Seccomp"],
        "seccomp_mode_name": {0: "disabled", 1: "strict", 2: "filter"}.get(fields["Seccomp"], "unknown"),
        "seccomp_filter_count": fields["Seccomp_filters"],
    },
    "provenance": {
        "env_controls_or_records_policy": any(term in env for term in terms),
        "serial_harness_controls_or_records_policy": any(term in harness for term in terms),
    },
}
OUT.parent.mkdir(parents=True, exist_ok=True)
OUT.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
print(OUT)
