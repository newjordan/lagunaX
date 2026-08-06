#!/usr/bin/env python3
"""Audit CPU-affinity controls and provenance for the canonical serial benchmark."""
import json
import os
import re
import subprocess
from pathlib import Path

root = Path(__file__).resolve().parents[1]
env_text = (root / "env.sh").read_text()
bench_text = (root / "scripts/bench-serial.sh").read_text()

cmd = ["bash", "-lc", 'source ./env.sh >/dev/null 2>&1; "$LX_LLAMA_BENCH" --help 2>&1']
help_text = subprocess.run(cmd, cwd=root, text=True, capture_output=True, check=True).stdout
mask = re.search(r"-C, --cpu-mask <hex,hex>\s+\(default: ([^)]+)\)", help_text)
strict = re.search(r"--cpu-strict <0\|1>\s+\(default: ([^)]+)\)", help_text)
if not mask or not strict:
    raise SystemExit("live llama-bench CPU-affinity contract not found")

combined = env_text + "\n" + bench_text
configured = bool(re.search(r"(?:--cpu-mask|-C\s|--cpu-strict|CPU_MASK|CPU_STRICT)", combined))
metrics_records = bool(re.search(r'["\'](?:cpu_mask|cpu_strict|affinity)["\']', bench_text, re.I))
allowed = sorted(os.sched_getaffinity(0))
artifact = {
    "angle": "host CPU affinity and strict placement",
    "live_contract": {"cpu_mask_default": mask.group(1), "cpu_strict_default": int(strict.group(1))},
    "process_allowed_cpus": allowed,
    "process_allowed_cpu_count": len(allowed),
    "laguna_configures_affinity": configured,
    "metrics_record_affinity": metrics_records,
}
assert artifact["live_contract"] == {"cpu_mask_default": "0x0", "cpu_strict_default": 0}
assert configured is False
assert metrics_records is False
out = root / "results/cpu-affinity-placement-policy-audit-20260807.json"
out.write_text(json.dumps(artifact, indent=2) + "\n")
print(out)
