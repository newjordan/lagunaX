#!/usr/bin/env python3
"""Audit PCIe AER error counters and Laguna benchmark provenance."""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEVICE = "0000:0d:00.0"
OUT = ROOT / "results/accelerator-pcie-aer-error-policy-audit-20260807.json"


def read(path: Path) -> str | None:
    try:
        return path.read_text().strip()
    except OSError:
        return None


def aer_state(path: Path) -> dict[str, object]:
    counters = {}
    for name in ("aer_dev_correctable", "aer_dev_nonfatal", "aer_dev_fatal"):
        value = read(path / name)
        if value is not None:
            counters[name] = {
                "raw": value,
                "total": sum(int(line.rsplit(maxsplit=1)[-1]) for line in value.splitlines()),
            }
    return {"bdf": path.name, "counters": counters}


endpoint = (Path("/sys/bus/pci/devices") / DEVICE).resolve()
hierarchy = []
path = endpoint
while path.name.startswith("0000:"):
    hierarchy.append(aer_state(path))
    path = path.parent

env_text = (ROOT / "env.sh").read_text(errors="replace")
harness_text = (ROOT / "scripts/bench-serial.sh").read_text(errors="replace")
source = (env_text + "\n" + harness_text).lower()
needles = ("aer_dev_correctable", "aer_dev_nonfatal", "aer_dev_fatal", "pcie aer", "aer error")
report = {
    "generated_utc": datetime.now(timezone.utc).isoformat(),
    "angle": "accelerator-pcie-aer-error-policy",
    "live": {"device": DEVICE, "pcie_hierarchy": hierarchy},
    "laguna": {
        "configures_or_rejects_aer_error_state": any(term in source for term in needles),
        "records_aer_error_counters": any(term in harness_text.lower() for term in needles),
    },
}
assert hierarchy
assert any(node["counters"] for node in hierarchy), "no PCIe AER counters exported in hierarchy"
assert not report["laguna"]["configures_or_rejects_aer_error_state"]
assert not report["laguna"]["records_aer_error_counters"]
OUT.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
print(OUT)
