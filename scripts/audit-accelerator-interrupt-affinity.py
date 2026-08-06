#!/usr/bin/env python3
"""Audit accelerator interrupt routing and Laguna benchmark provenance."""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEVICE = "0000:0d:00.0"
OUT = ROOT / "results" / "accelerator-interrupt-affinity-audit-20260807.json"


def read(path: Path) -> str | None:
    try:
        return path.read_text().strip()
    except OSError:
        return None


irq_dir = Path(f"/sys/bus/pci/devices/{DEVICE}/msi_irqs")
irqs = sorted((p.name for p in irq_dir.iterdir()), key=int) if irq_dir.is_dir() else []
interrupt_lines = Path("/proc/interrupts").read_text().splitlines()
records = []
for irq in irqs:
    line = next((x.strip() for x in interrupt_lines if x.lstrip().startswith(f"{irq}:")), None)
    proc = Path("/proc/irq") / irq
    records.append({
        "irq": int(irq),
        "configured_affinity_list": read(proc / "smp_affinity_list"),
        "effective_affinity_list": read(proc / "effective_affinity_list"),
        "numa_node": read(proc / "node"),
        "interrupts_line": line,
    })

env_text = (ROOT / "env.sh").read_text()
harness_text = (ROOT / "scripts" / "bench-serial.sh").read_text()
source = (env_text + "\n" + harness_text).lower()
needles = ("smp_affinity", "effective_affinity", "msi_irqs", "/proc/interrupts", "irqbalance")
report = {
    "generated_utc": datetime.now(timezone.utc).isoformat(),
    "scope": "accelerator MSI interrupt routing and affinity",
    "live": {
        "device": DEVICE,
        "online_cpus": read(Path("/sys/devices/system/cpu/online")),
        "default_smp_affinity": read(Path("/proc/irq/default_smp_affinity")),
        "msi_irq_count": len(irqs),
        "interrupts": records,
    },
    "laguna": {
        "canonical_environment": "env.sh",
        "serial_harness": "scripts/bench-serial.sh",
        "configures_accelerator_interrupt_affinity": any(x in source for x in needles),
        "records_accelerator_interrupt_affinity": any(x in harness_text.lower() for x in needles),
    },
}
OUT.write_text(json.dumps(report, indent=2) + "\n")
print(OUT)
