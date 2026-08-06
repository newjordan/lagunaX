#!/usr/bin/env python3
"""Audit IOMMU/DMA-remapping state and Laguna benchmark provenance."""
from __future__ import annotations

import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEVICE = "0000:0d:00.0"
OUT = ROOT / "results" / "accelerator-iommu-dma-remapping-audit-20260807.json"


def read(path: str) -> str | None:
    try:
        return Path(path).read_text().strip()
    except OSError:
        return None


cmdline = read("/proc/cmdline")
group_link = Path(f"/sys/bus/pci/devices/{DEVICE}/iommu_group")
try:
    iommu_group = str(group_link.resolve(strict=True))
except OSError:
    iommu_group = None
journal = subprocess.run(
    ["journalctl", "-k", "-b", "--no-pager"],
    text=True, capture_output=True, check=False,
).stdout.splitlines()
iommu_log = [line for line in journal if any(x in line.lower() for x in ("iommu", "swiotlb", "dma remap"))]
env_text = (ROOT / "env.sh").read_text()
harness_text = (ROOT / "scripts" / "bench-serial.sh").read_text()
source = (env_text + "\n" + harness_text).lower()
needles = ("iommu", "swiotlb", "dma_remap", "dma-remap")

report = {
    "generated_utc": datetime.now(timezone.utc).isoformat(),
    "scope": "accelerator IOMMU and DMA-remapping state",
    "live": {
        "device": DEVICE,
        "kernel_cmdline": cmdline,
        "iommu_group": iommu_group,
        "iommu_group_present": iommu_group is not None,
        "kernel_iommu_log": iommu_log,
    },
    "laguna": {
        "canonical_environment": "env.sh",
        "serial_harness": "scripts/bench-serial.sh",
        "configures_iommu_or_dma_remapping": any(x in source for x in needles),
        "records_iommu_or_dma_remapping": any(x in harness_text.lower() for x in needles),
    },
}
OUT.write_text(json.dumps(report, indent=2) + "\n")
print(OUT)
