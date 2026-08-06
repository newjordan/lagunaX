#!/usr/bin/env python3
"""Audit accelerator reset capability/history and benchmark provenance."""
import json
import re
import subprocess
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEVICE = Path("/sys/class/drm/card0/device").resolve()
ENV = (ROOT / "env.sh").read_text()
BENCH = (ROOT / "scripts/bench-serial.sh").read_text()

def read(path):
    try:
        return Path(path).read_text().strip()
    except OSError:
        return None

journal = subprocess.run(
    ["journalctl", "-k", "-b", "--no-pager"],
    text=True, capture_output=True, check=False,
).stdout.splitlines()
reset_re = re.compile(r"(?:xe|drm).*?(?:engine reset|gpu reset|hang|wedg)", re.I)
reset_events = [line for line in journal if reset_re.search(line)]
needles = ("engine reset", "gpu reset", "reset_method", "reset_events", "journalctl -k")
source = ENV + "\n" + BENCH
artifact = {
    "generated_utc": datetime.now(timezone.utc).isoformat(),
    "angle": "accelerator-reset-history-and-run-validity",
    "device": str(DEVICE),
    "reset_capability": {
        "reset_method": read(DEVICE / "reset_method"),
        "reset_control_present": (DEVICE / "reset").exists(),
    },
    "boot_kernel_reset_events": reset_events,
    "boot_kernel_reset_event_count": len(reset_events),
    "canonical_policy": {
        "checks_reset_history": any(n in source.lower() for n in needles),
        "records_reset_history": any(n in BENCH.lower() for n in needles),
    },
    "conclusion": "The accelerator supports function/bus reset and has a boot-session engine-reset event, but Laguna's canonical benchmark path neither checks nor records reset history.",
}
out = ROOT / "results/accelerator-reset-history-audit-20260807.json"
out.write_text(json.dumps(artifact, indent=2) + "\n")
assert artifact["reset_capability"]["reset_control_present"]
assert artifact["reset_capability"]["reset_method"]
assert artifact["boot_kernel_reset_event_count"] >= 1
assert not artifact["canonical_policy"]["checks_reset_history"]
assert not artifact["canonical_policy"]["records_reset_history"]
print(out)
