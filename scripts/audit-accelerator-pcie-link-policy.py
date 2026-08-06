#!/usr/bin/env python3
"""Audit negotiated PCIe link geometry and ASPM provenance for Laguna."""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEVICE = "0000:0d:00.0"
OUT = ROOT / "results" / "accelerator-pcie-link-policy-audit-20260807.json"


def read(path: Path) -> str | None:
    try:
        return path.read_text().strip()
    except OSError:
        return None


def link_state(device: str) -> dict[str, str | None]:
    base = Path("/sys/bus/pci/devices") / device
    return {
        key: read(base / key)
        for key in ("current_link_speed", "current_link_width", "max_link_speed", "max_link_width")
    }


device_path = (Path("/sys/bus/pci/devices") / DEVICE).resolve()
upstream = device_path.parent.name
cmdline = read(Path("/proc/cmdline"))
aspm_policy = read(Path("/sys/module/pcie_aspm/parameters/policy"))
env_text = (ROOT / "env.sh").read_text()
harness_text = (ROOT / "scripts" / "bench-serial.sh").read_text()
source = (env_text + "\n" + harness_text).lower()
needles = ("pcie_aspm", "aspm", "current_link_speed", "current_link_width", "max_link_speed", "max_link_width")

report = {
    "generated_utc": datetime.now(timezone.utc).isoformat(),
    "scope": "accelerator PCIe negotiated link geometry and ASPM policy",
    "live": {
        "device": DEVICE,
        "device_link": link_state(DEVICE),
        "upstream_bridge": upstream,
        "upstream_link": link_state(upstream),
        "kernel_cmdline": cmdline,
        "aspm_policy": aspm_policy,
        "kernel_aspm_disabled": bool(cmdline and "pcie_aspm=off" in cmdline.split()),
    },
    "laguna": {
        "canonical_environment": "env.sh",
        "serial_harness": "scripts/bench-serial.sh",
        "configures_pcie_link_or_aspm": any(x in source for x in needles),
        "records_pcie_link_or_aspm": any(x in harness_text.lower() for x in needles),
    },
}
OUT.write_text(json.dumps(report, indent=2) + "\n")
print(OUT)
