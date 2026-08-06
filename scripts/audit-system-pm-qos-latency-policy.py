#!/usr/bin/env python3
"""Audit system PM-QoS latency constraints and Laguna provenance."""
import json
from pathlib import Path

root = Path(__file__).resolve().parents[1]


def read(path: Path):
    try:
        return path.read_text().strip()
    except OSError:
        return None


cpu_latency = Path("/sys/devices/system/cpu/cpu0/power/pm_qos_resume_latency_us")
latency_devices = []
for path in sorted(Path("/sys/devices").glob("**/power/pm_qos_resume_latency_us")):
    latency_devices.append({"path": str(path), "microseconds": read(path)})

sources = {p: (root / p).read_text() for p in ("env.sh", "scripts/bench-serial.sh")}
needles = ("cpu_dma_latency", "pm_qos_resume_latency_us", "PM_QOS", "pm_qos")
artifact = {
    "cpu0_resume_latency_us": read(cpu_latency),
    "device_resume_latency_constraints": latency_devices,
    "cpu_dma_latency_device_present": Path("/dev/cpu_dma_latency").exists(),
    "laguna_source_mentions": {
        name: {needle: needle in text for needle in needles}
        for name, text in sources.items()
    },
    "conclusion": (
        "The host exports PM-QoS latency controls, but Laguna's canonical serial path "
        "neither requests nor records a latency constraint."
    ),
}
out = root / "results/system-pm-qos-latency-policy-audit-20260807.json"
out.write_text(json.dumps(artifact, indent=2) + "\n")
assert artifact["cpu0_resume_latency_us"] is not None or latency_devices
assert all(
    not found
    for source in artifact["laguna_source_mentions"].values()
    for found in source.values()
)
print(out)
