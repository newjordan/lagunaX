#!/usr/bin/env python3
"""Audit whether benchmark artifacts preserve GPU operating state."""
from pathlib import Path
import json

ROOT = Path(__file__).resolve().parents[1]
files = sorted((ROOT / "results").rglob("*.json"))
records = 0
fields = {"temperature": 0, "frequency": 0, "power": 0, "throttle": 0}
needles = {
    "temperature": ("temperature", "temp_c", "gpu_temp"),
    "frequency": ("frequency", "clock_mhz", "gpu_clock"),
    "power": ("power_w", "power_watts", "gpu_power"),
    "throttle": ("throttle", "throttled", "throttling"),
}

def walk(x):
    if isinstance(x, dict):
        yield x
        for value in x.values(): yield from walk(value)
    elif isinstance(x, list):
        for value in x: yield from walk(value)

for path in files:
    try: payload = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError, UnicodeDecodeError): continue
    records += 1
    keys = {str(k).lower() for obj in walk(payload) for k in obj}
    for kind, aliases in needles.items():
        fields[kind] += int(any(alias in keys for alias in aliases))

out = ROOT / "results" / "device-operating-state-audit-20260807.txt"
out.write_text("\n".join([
    f"json_artifacts={records}",
    *(f"artifacts_with_{kind}_field={count}" for kind, count in fields.items()),
    f"operating_state_complete={str(all(fields.values())).lower()}",
]) + "\n")
assert records > 0
assert fields["frequency"] == fields["power"] == fields["throttle"] == 0, fields
assert out.read_text().endswith("\n") and not any(line.endswith(" ") for line in out.read_text().splitlines())
print(out)
