#!/usr/bin/env python3
"""Audit kernel clocksource selection and Laguna benchmark provenance."""
import json
import re
from datetime import datetime, timezone
from pathlib import Path

root = Path(__file__).resolve().parents[1]
env_text = (root / "env.sh").read_text()
bench_text = (root / "scripts/bench-serial.sh").read_text()
clock_root = Path("/sys/devices/system/clocksource/clocksource0")

def read(name: str) -> str | None:
    path = clock_root / name
    return path.read_text().strip() if path.exists() else None

current = read("current_clocksource")
available_text = read("available_clocksource") or ""
available = available_text.split()
cmdline = Path("/proc/cmdline").read_text().strip()
clocksource_overrides = [
    token for token in cmdline.split()
    if token.startswith(("clocksource=", "tsc=", "hpet=", "nohpet"))
]
combined = env_text + "\n" + bench_text
configures = bool(re.search(r"(?:current_clocksource|clocksource=|tsc=|hpet=|nohpet)", combined, re.I))
records = bool(re.search(r'["\'](?:clocksource|current_clocksource|available_clocksource)["\']', bench_text, re.I))

artifact = {
    "captured_at_utc": datetime.now(timezone.utc).isoformat(),
    "angle": "kernel timekeeping clocksource selection and benchmark provenance",
    "kernel": {
        "current_clocksource": current,
        "available_clocksources": available,
        "command_line_overrides": clocksource_overrides,
    },
    "laguna": {
        "configures_clocksource": configures,
        "records_clocksource": records,
    },
}
assert current
assert current in available
out = root / "results/kernel-clocksource-policy-audit-20260807.json"
out.write_text(json.dumps(artifact, indent=2) + "\n")
print(out)
