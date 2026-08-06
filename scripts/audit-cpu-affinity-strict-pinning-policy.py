#!/usr/bin/env python3
"""Audit llama-bench CPU affinity/strict-pinning controls and Laguna provenance."""
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
binary = env.get(
    "LX_LLAMA_BENCH",
    str(ROOT / "baseline/tip-binary-backup-20260730T141542Z/llama-bench"),
)
probe = subprocess.run(
    [binary, "--help"], text=True, stdout=subprocess.PIPE,
    stderr=subprocess.STDOUT, check=True,
).stdout
mask = re.search(r"(?:-C, )?--cpu-mask <hex,hex>\s+\(default: ([^)]+)\)", probe)
strict = re.search(r"--cpu-strict <0\|1>\s+\(default: (\d+)\)", probe)
if not mask or not strict:
    raise SystemExit("live llama-bench help lacks expected CPU-affinity contract")
passes_mask = bool(re.search(r"(?:^|\s)(?:-C|--cpu-mask)(?:\s|$)", bench_text, re.M))
passes_strict = bool(re.search(r"(?:^|\s)--cpu-strict(?:\s|$)", bench_text, re.M))
records_mask = bool(re.search(r'["\'](?:cpu_mask|cpu-mask)["\']\s*:', bench_text))
records_strict = bool(re.search(r'["\'](?:cpu_strict|cpu-strict)["\']\s*:', bench_text))
env_affinity = bool(re.search(r"(?:CPU_MASK|CPU_STRICT|GOMP_CPU_AFFINITY|KMP_AFFINITY)", env_text))
artifact = {
    "collected_at": datetime.now(timezone.utc).isoformat(),
    "binary": binary,
    "controls": {
        "cpu_mask": {"option": "--cpu-mask", "default": mask.group(1)},
        "cpu_strict": {"option": "--cpu-strict", "accepted_values": [0, 1], "default": int(strict.group(1))},
    },
    "laguna": {
        "environment_configures_affinity": env_affinity,
        "serial_harness_passes_cpu_mask": passes_mask,
        "serial_harness_passes_cpu_strict": passes_strict,
        "serial_metrics_record_cpu_mask": records_mask,
        "serial_metrics_record_cpu_strict": records_strict,
    },
}
out = ROOT / "results/cpu-affinity-strict-pinning-policy-audit-20260807.json"
out.write_text(json.dumps(artifact, indent=2) + "\n")
assert artifact["controls"]["cpu_mask"]["default"] == "0x0"
assert artifact["controls"]["cpu_strict"]["default"] == 0
assert artifact["laguna"] == {
    "environment_configures_affinity": False,
    "serial_harness_passes_cpu_mask": False,
    "serial_harness_passes_cpu_strict": False,
    "serial_metrics_record_cpu_mask": False,
    "serial_metrics_record_cpu_strict": False,
}
print(out)
