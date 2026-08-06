#!/usr/bin/env python3
"""Audit IRQ default/effective affinity and Laguna provenance."""
import json
import re
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ENV = ROOT / "env.sh"
HARNESS = ROOT / "scripts/bench-serial.sh"
OUT = ROOT / "results/kernel-irq-affinity-policy-audit-20260807.json"


def read_text(path: str) -> str:
    try:
        return Path(path).read_text().strip()
    except (FileNotFoundError, PermissionError) as exc:
        raise RuntimeError(f"cannot read {path}: {exc}") from exc


def cpu_list_count(spec: str) -> int:
    total = 0
    for part in spec.split(","):
        bounds = part.split("-", 1)
        total += 1 if len(bounds) == 1 else int(bounds[1]) - int(bounds[0]) + 1
    return total


possible = read_text("/sys/devices/system/cpu/possible")
default_affinity = read_text("/proc/irq/default_smp_affinity")
irqs = []
for entry in sorted(Path("/proc/irq").iterdir(), key=lambda p: int(p.name) if p.name.isdigit() else -1):
    if not entry.name.isdigit():
        continue
    effective = entry / "effective_affinity_list"
    configured = entry / "smp_affinity_list"
    if effective.exists() and configured.exists():
        irqs.append({
            "irq": int(entry.name),
            "configured_cpu_list": read_text(str(configured)),
            "effective_cpu_list": read_text(str(effective)),
        })

env_text = ENV.read_text(errors="replace")
harness_text = HARNESS.read_text(errors="replace")
source = env_text + "\n" + harness_text
pattern = re.compile(r"default_smp_affinity|smp_affinity(?:_list)?|effective_affinity", re.I)
unique_effective = sorted({item["effective_cpu_list"] for item in irqs})
report = {
    "angle": "kernel-irq-affinity-policy",
    "generated_at": datetime.now(timezone.utc).isoformat(),
    "live": {
        "possible_cpu_list": possible,
        "possible_cpu_count": cpu_list_count(possible),
        "default_smp_affinity_hex": default_affinity,
        "irq_count_with_affinity": len(irqs),
        "unique_effective_cpu_lists": unique_effective,
        "irqs": irqs,
    },
    "laguna": {
        "env_or_harness_controls_irq_affinity": bool(pattern.search(source)),
        "serial_harness_records_irq_affinity": bool(pattern.search(harness_text)),
    },
}
assert report["live"]["possible_cpu_count"] > 0
assert report["live"]["irq_count_with_affinity"] > 0
assert unique_effective
assert not report["laguna"]["env_or_harness_controls_irq_affinity"]
assert not report["laguna"]["serial_harness_records_irq_affinity"]
OUT.parent.mkdir(parents=True, exist_ok=True)
OUT.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
print(OUT)
