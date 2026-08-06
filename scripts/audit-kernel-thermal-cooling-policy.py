#!/usr/bin/env python3
"""Audit live thermal-zone and cooling-device policy and benchmark provenance."""
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


zones = []
for path in sorted(Path("/sys/class/thermal").glob("thermal_zone*")):
    zones.append({
        "name": path.name,
        "type": read(path / "type"),
        "temp_millicelsius": read(path / "temp"),
        "policy": read(path / "policy"),
    })

cooling = []
for path in sorted(Path("/sys/class/thermal").glob("cooling_device*")):
    cooling.append({
        "name": path.name,
        "type": read(path / "type"),
        "current_state": read(path / "cur_state"),
        "max_state": read(path / "max_state"),
    })

TOKENS = ("thermal_zone", "cooling_device", "trip_point", "cur_state", "fan")
report = {
    "angle": "kernel-thermal-cooling-policy",
    "generated_at": datetime.now(timezone.utc).isoformat(),
    "live": {
        "thermal_zones": zones,
        "cooling_devices": cooling,
        "nonzero_cooling_devices": [d for d in cooling if d["current_state"] not in (None, "0")],
    },
    "laguna": {
        "env_policy_hits": [token for token in TOKENS if token in ENV],
        "bench_policy_hits": [token for token in TOKENS if token in BENCH],
        "metrics_record_policy": any(f'"{token}"' in BENCH for token in TOKENS),
    },
}
out = ROOT / "benchmark/results/kernel-thermal-cooling-policy-audit-20260807.json"
out.parent.mkdir(parents=True, exist_ok=True)
out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
print(out)
