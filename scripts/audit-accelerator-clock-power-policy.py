#!/usr/bin/env python3
"""Read-only audit of accelerator clock and power-limit policy provenance."""
import json
import shutil
import subprocess
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "benchmark" / "results" / "accelerator-clock-power-policy-audit-20260807.json"
ENV = (ROOT / "env.sh").read_text(errors="replace")
BENCH = (ROOT / "scripts" / "bench-serial.sh").read_text(errors="replace")
TOOLS = ("nvidia-smi", "rocm-smi", "xpu-smi")
KEYWORDS = ("clock", "power_limit", "power-limit", "nvidia-smi", "rocm-smi", "xpu-smi")

def mentions(text: str) -> list[str]:
    lower = text.lower()
    return [key for key in KEYWORDS if key in lower]

def probe(name: str) -> dict:
    path = shutil.which(name)
    result = {"path": path, "available": path is not None}
    if path:
        completed = subprocess.run([path, "--help"], text=True, capture_output=True, timeout=10)
        result.update(returncode=completed.returncode)
    return result

artifact = {
    "audit": "accelerator_clock_power_policy",
    "timestamp_utc": datetime.now(timezone.utc).isoformat(),
    "read_only": True,
    "management_tools": {name: probe(name) for name in TOOLS},
    "laguna_provenance": {
        "env_mentions": mentions(ENV),
        "serial_harness_mentions": mentions(BENCH),
        "controls_or_records_clock_power_policy": bool(mentions(ENV) or mentions(BENCH)),
    },
    "checkpoint": json.loads((ROOT / "results" / "mandatory-primary-evidence-checkpoint-20260807.json").read_text()),
}
OUT.parent.mkdir(parents=True, exist_ok=True)
OUT.write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n")
print(OUT)
