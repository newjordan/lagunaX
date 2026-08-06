#!/usr/bin/env python3
"""Audit accelerator power-cap and thermal provenance in Laguna benchmarks."""
import json
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DRM = Path("/sys/class/drm/card0/device")
HWMON = next(iter((DRM / "hwmon").glob("hwmon*")), None)

def read(path):
    try:
        return path.read_text().strip()
    except OSError:
        return None

def mentions(text, needles):
    low = text.lower()
    return any(n in low for n in needles)

env = (ROOT / "env.sh").read_text()
harness = (ROOT / "scripts/bench-serial.sh").read_text()
temps = {}
if HWMON:
    for label_path in sorted(HWMON.glob("temp*_label")):
        stem = label_path.name[:-6]
        temps[read(label_path) or stem] = {
            "input_millicelsius": int(read(HWMON / f"{stem}_input")),
            "critical_millicelsius": int(read(HWMON / f"{stem}_crit")),
        }
report = {
    "generated_utc": datetime.now(timezone.utc).isoformat(),
    "device_pci": DRM.resolve().name,
    "hwmon_name": read(HWMON / "name") if HWMON else None,
    "power_cap_microwatts": int(read(HWMON / "power1_cap")) if HWMON else None,
    "power_critical_microwatts": int(read(HWMON / "power1_crit")) if HWMON else None,
    "power_cap_interval_ms": int(read(HWMON / "power1_cap_interval")) if HWMON else None,
    "temperatures": temps,
    "laguna_configures_power_cap": mentions(env + harness, ["power1_cap", "power_limit", "power-limit"]),
    "laguna_records_power_or_temperature": mentions(harness, ["power1_cap", "energy1_input", "temp1_input", "temperature"]),
}
out = ROOT / "benchmark/results/accelerator-power-thermal-policy-audit-20260807.json"
out.parent.mkdir(parents=True, exist_ok=True)
out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
print(out)
