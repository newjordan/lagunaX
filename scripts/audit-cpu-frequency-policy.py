#!/usr/bin/env python3
"""Audit CPU frequency/governor policy and benchmark provenance."""
import json
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BENCH = (ROOT / "scripts/bench-serial.sh").read_text()
ENV = (ROOT / "env.sh").read_text()
CPU = Path("/sys/devices/system/cpu/cpu0/cpufreq")


def read(name):
    path = CPU / name
    try:
        return path.read_text().strip()
    except OSError:
        return None

fields = {
    name: read(name)
    for name in (
        "scaling_driver", "scaling_governor", "scaling_min_freq",
        "scaling_max_freq", "cpuinfo_min_freq", "cpuinfo_max_freq",
    )
}
needles = ("scaling_governor", "scaling_min_freq", "scaling_max_freq", "cpuinfo_max_freq")
payload = {
    "generated_utc": datetime.now(timezone.utc).isoformat(),
    "cpu0_cpufreq": fields,
    "canonical_policy": {
        "env_has_cpufreq_control": any(x in ENV for x in needles),
        "harness_has_cpufreq_control": any(x in BENCH for x in needles),
        "metrics_record_governor": '"scaling_governor"' in BENCH,
        "metrics_record_frequency_bounds": any(f'"{x}"' in BENCH for x in needles[1:]),
    },
}
out = ROOT / "results/cpu-frequency-policy-audit-20260807.json"
out.write_text(json.dumps(payload, indent=2) + "\n")
print(out)
assert fields["scaling_driver"] is not None
assert fields["scaling_governor"] is not None
assert not any(payload["canonical_policy"].values())
