#!/usr/bin/env python3
"""Audit CPU vulnerability mitigations and Laguna benchmark provenance."""
import json
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ENV = (ROOT / "env.sh").read_text(errors="replace")
BENCH = (ROOT / "scripts/bench-serial.sh").read_text(errors="replace")


def read(path):
    try:
        return path.read_text().strip()
    except OSError:
        return None


vulnerability_dir = Path("/sys/devices/system/cpu/vulnerabilities")
vulnerabilities = {
    path.name: read(path)
    for path in sorted(vulnerability_dir.glob("*"))
    if path.is_file()
}
cmdline = read(Path("/proc/cmdline")) or ""
source = ENV + "\n" + BENCH
control_terms = (
    "mitigations=", "spectre_v2=", "spec_store_bypass_disable=", "l1tf=",
    "mds=", "tsx_async_abort=", "retbleed=", "gather_data_sampling=",
    "reg_file_data_sampling=",
)
metric_terms = (
    "cpu_vulnerabilities", "cpu_mitigation_state", "kernel_mitigation_controls",
)
report = {
    "generated_utc": datetime.now(timezone.utc).isoformat(),
    "angle": "cpu-speculation-mitigation-policy",
    "live_policy": {
        "vulnerabilities": vulnerabilities,
        "kernel_cmdline_mitigation_controls": [
            term for term in control_terms if term in cmdline
        ],
    },
    "canonical_policy": {
        "env_or_harness_controls_mitigations": [
            term for term in control_terms if term in source
        ],
        "metrics_record_mitigation_state": [
            term for term in metric_terms if f'\"{term}\"' in BENCH
        ],
    },
}

assert vulnerabilities
assert all(value is not None for value in vulnerabilities.values())
assert not report["canonical_policy"]["env_or_harness_controls_mitigations"]
assert not report["canonical_policy"]["metrics_record_mitigation_state"]
out = ROOT / "results/cpu-speculation-mitigation-policy-audit-20260807.json"
out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
print(out)
