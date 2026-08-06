#!/usr/bin/env python3
"""Audit CPU speculative-execution vulnerability mitigations and Laguna provenance."""
import json
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
VULNERABILITIES = Path("/sys/devices/system/cpu/vulnerabilities")


def read(path):
    try:
        return path.read_text(errors="replace").strip()
    except OSError:
        return None


status = {
    path.name: read(path)
    for path in sorted(VULNERABILITIES.iterdir())
    if path.is_file()
}
cmdline = read(Path("/proc/cmdline")) or ""
mitigation_terms = (
    "mitigations=", "spectre_v2=", "spec_store_bypass_disable=", "l1tf=",
    "mds=", "tsx_async_abort=", "retbleed=", "gather_data_sampling=",
    "reg_file_data_sampling=", "srbds=",
)
sources = {
    "env.sh": (ROOT / "env.sh").read_text(errors="replace"),
    "scripts/bench-serial.sh": (ROOT / "scripts/bench-serial.sh").read_text(errors="replace"),
}
report = {
    "generated_utc": datetime.now(timezone.utc).isoformat(),
    "angle": "cpu-speculative-execution-mitigation-policy",
    "live_policy": {
        "vulnerability_status": status,
        "kernel_cmdline": cmdline,
        "explicit_cmdline_controls": [term for term in mitigation_terms if term in cmdline],
        "not_affected_count": sum(value == "Not affected" for value in status.values()),
        "mitigated_count": sum(value.startswith("Mitigation:") for value in status.values()),
        "vulnerable_count": sum("Vulnerable" in value for value in status.values()),
    },
    "canonical_policy": {
        path: [term for term in mitigation_terms if term.lower() in text.lower()]
        for path, text in sources.items()
    },
}
assert status
assert report["live_policy"]["vulnerable_count"] == 0
assert not report["canonical_policy"]["env.sh"]
assert not report["canonical_policy"]["scripts/bench-serial.sh"]
out = ROOT / "results/cpu-speculative-execution-mitigation-policy-audit-20260807.json"
out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
print(out)
