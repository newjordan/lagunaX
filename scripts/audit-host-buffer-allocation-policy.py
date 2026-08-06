#!/usr/bin/env python3
"""Audit llama-bench --no-host policy and Laguna provenance."""
import json
import os
import re
import subprocess
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
env_text = (ROOT / "env.sh").read_text()
bench_text = (ROOT / "scripts/bench-serial.sh").read_text()
env = os.environ.copy()
probe = subprocess.run(
    [env.get("LX_LLAMA_BENCH", str(ROOT / "baseline/tip-binary-backup-20260730T141542Z/llama-bench")), "--help"],
    text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, check=True,
).stdout
match = re.search(r"--no-host <0\|1>\s+\(default: (\d+)\)", probe)
if not match:
    raise SystemExit("live llama-bench help lacks expected --no-host contract")
configured = bool(re.search(r"(?:^|\s)(?:-no-host|--no-host)(?:\s|$)", bench_text, re.M))
recorded = bool(re.search(r'["\'](?:no_host|no-host|host_buffer)["\']\s*:', bench_text))
env_configured = bool(re.search(r"(?:NO_HOST|NOHOST|GGML_NO_HOST)", env_text))
artifact = {
    "collected_at": datetime.now(timezone.utc).isoformat(),
    "control": "--no-host",
    "supported": True,
    "accepted_values": [0, 1],
    "default": int(match.group(1)),
    "semantics": "disable host buffer allocation",
    "laguna": {
        "environment_configures_control": env_configured,
        "serial_harness_passes_control": configured,
        "serial_metrics_record_control": recorded,
    },
}
out = ROOT / "results/host-buffer-allocation-policy-audit-20260807.json"
out.write_text(json.dumps(artifact, indent=2) + "\n")
assert artifact["default"] == 0
assert artifact["laguna"] == {
    "environment_configures_control": False,
    "serial_harness_passes_control": False,
    "serial_metrics_record_control": False,
}
print(out)
