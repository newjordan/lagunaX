#!/usr/bin/env python3
"""Audit Linux pressure-stall information and Laguna benchmark provenance."""
import json
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ENV = (ROOT / "env.sh").read_text(errors="replace")
BENCH = (ROOT / "scripts/bench-serial.sh").read_text(errors="replace")


def parse_pressure(path: Path) -> dict:
    result = {}
    for line in path.read_text(errors="replace").splitlines():
        fields = line.split()
        result[fields[0]] = {
            key: (int(value) if key == "total" else float(value))
            for key, value in (field.split("=", 1) for field in fields[1:])
        }
    return result


pressure = {
    resource: parse_pressure(Path(f"/proc/pressure/{resource}"))
    for resource in ("cpu", "memory", "io")
}
source = ENV + "\n" + BENCH
terms = ("/proc/pressure", "pressure/cpu", "pressure/memory", "pressure/io", "psi")
report = {
    "generated_utc": datetime.now(timezone.utc).isoformat(),
    "angle": "linux-pressure-stall-information-provenance",
    "live_pressure": pressure,
    "canonical_policy": {
        "env_or_harness_reads_psi": [term for term in terms if term in source.lower()],
        "metrics_record_psi": [term for term in terms if term in BENCH.lower()],
    },
}
assert set(pressure) == {"cpu", "memory", "io"}
assert "some" in pressure["cpu"]
assert all({"some", "full"} <= set(pressure[name]) for name in ("memory", "io"))
assert not report["canonical_policy"]["env_or_harness_reads_psi"]
assert not report["canonical_policy"]["metrics_record_psi"]
out = ROOT / "results/linux-pressure-stall-information-audit-20260807.json"
out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
print(out)
