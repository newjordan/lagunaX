#!/usr/bin/env python3
"""Audit accelerator device visibility and selection for the canonical serial run."""
import json
import os
import re
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ENV = ROOT / "env.sh"
BENCH = ROOT / "scripts/bench-serial.sh"
OUT = ROOT / "results/accelerator-device-selection-audit-20260807.json"


def exported_default(text: str, name: str):
    m = re.search(rf'^export {re.escape(name)}="\$\{{{re.escape(name)}:-([^}}]+)\}}"', text, re.M)
    return m.group(1) if m else None


env_text = ENV.read_text()
bench_text = BENCH.read_text()
cards = []
for card in sorted(p for p in Path("/sys/class/drm").iterdir() if re.fullmatch(r"card[0-9]+", p.name)):
    device = card / "device"
    cards.append({
        "card": card.name,
        "pci_address": device.resolve().name,
        "vendor": (device / "vendor").read_text().strip() if (device / "vendor").exists() else None,
        "device": (device / "device").read_text().strip() if (device / "device").exists() else None,
        "tiles": len(list(device.glob("tile*"))),
    })

selector = exported_default(env_text, "ONEAPI_DEVICE_SELECTOR")
affinity = exported_default(env_text, "ZE_AFFINITY_MASK")
result = {
    "audit": "accelerator-device-selection",
    "env_defaults": {"ONEAPI_DEVICE_SELECTOR": selector, "ZE_AFFINITY_MASK": affinity},
    "serial_harness_mentions": {
        "ONEAPI_DEVICE_SELECTOR": "ONEAPI_DEVICE_SELECTOR" in bench_text,
        "ZE_AFFINITY_MASK": "ZE_AFFINITY_MASK" in bench_text,
    },
    "drm_cards": cards,
    "selection_scope": "backend-wide GPU selection narrowed by Level Zero affinity index 0",
    "checks": {
        "selector_is_not_pci_stable": selector == "level_zero:gpu",
        "affinity_is_ordinal": affinity == "0",
        "multiple_drm_cards_visible": len(cards) > 1,
    },
}
OUT.write_text(json.dumps(result, indent=2) + "\n")
print(OUT)
