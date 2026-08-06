#!/usr/bin/env python3
"""Audit accelerator runtime-power and energy-telemetry policy/provenance."""
import json
from pathlib import Path

root = Path(__file__).resolve().parents[1]
drm = Path("/sys/class/drm/card0/device").resolve()
def read(path):
    try:
        return path.read_text().strip()
    except OSError:
        return None

hwmons = sorted((drm / "hwmon").glob("hwmon*"))
energy = []
for hw in hwmons:
    for p in sorted(hw.glob("energy*_input")):
        stem = p.name.removesuffix("_input")
        energy.append({"path": str(p), "label": read(hw / f"{stem}_label"), "microjoules": int(read(p))})

sources = {p: (root / p).read_text() for p in ("env.sh", "scripts/bench-serial.sh")}
needles = ("power/control", "runtime_status", "energy1_input", "energy2_input")
artifact = {
    "device": str(drm),
    "runtime_power": {
        "control": read(drm / "power/control"),
        "status": read(drm / "power/runtime_status"),
        "autosuspend_delay_ms": read(drm / "power/autosuspend_delay_ms"),
    },
    "energy_telemetry": energy,
    "laguna_source_mentions": {name: {n: n in text for n in needles} for name, text in sources.items()},
    "conclusion": "Accelerator runtime PM state and hardware energy counters are available but neither controlled nor recorded by Laguna's canonical serial path.",
}
out = root / "results/accelerator-runtime-power-policy-audit-20260807.json"
out.write_text(json.dumps(artifact, indent=2) + "\n")
assert artifact["runtime_power"]["control"] in {"auto", "on"}
assert artifact["runtime_power"]["status"] in {"active", "suspended", "suspending", "resuming", "unsupported"}
assert energy
assert all(not found for src in artifact["laguna_source_mentions"].values() for found in src.values())
print(out)
