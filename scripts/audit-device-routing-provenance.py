#!/usr/bin/env python3
"""Audit explicit llama-bench compute-device routing and Laguna provenance."""
import json
import os
import re
import subprocess
from datetime import datetime, timezone
from pathlib import Path

root = Path(__file__).resolve().parents[1]
env_text = (root / "env.sh").read_text()
bench_text = (root / "scripts/bench-serial.sh").read_text()
binary = os.environ.get("LX_LLAMA_BENCH", str(root / "baseline/tip-binary-backup-20260730T141542Z/llama-bench"))
help_text = subprocess.run([binary, "--help"], text=True, stdout=subprocess.PIPE,
                           stderr=subprocess.STDOUT, check=True).stdout
match = re.search(r"-dev, --device <dev0/dev1/\.\.\.>\s+\(default: ([^)]+)\)", help_text)
if not match:
    raise SystemExit("active llama-bench does not expose the expected --device contract")
list_run = subprocess.run([binary, "--list-devices"], text=True, stdout=subprocess.PIPE,
                          stderr=subprocess.STDOUT, check=True)
dev_lines = [line.strip() for line in list_run.stdout.splitlines() if line.strip()]
common = re.search(r"COMMON=\((.*?)\n\s*\)", bench_text, re.S)
common_text = common.group(1) if common else ""
explicit = bool(re.search(r"(^|\s)(-dev|--device)(\s|$)", common_text))
recorded = bool(re.search(r'["\']device["\']\s*:', bench_text, re.I))
payload = {
    "collected_at": datetime.now(timezone.utc).isoformat(),
    "binary": binary,
    "contract": {"supported": True, "default": match.group(1), "listed_devices": dev_lines},
    "laguna": {
        "canonical_selector": re.search(r'ONEAPI_DEVICE_SELECTOR[^\n]*', env_text).group(0),
        "explicit_device_argument": explicit,
        "effective_device_recorded_in_metrics": recorded,
    },
}
out = root / "results/device-routing-provenance-audit-20260807.json"
out.write_text(json.dumps(payload, indent=2) + "\n")
print(out)
