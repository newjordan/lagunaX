#!/usr/bin/env python3
"""Audit whether serial metrics preserve all performance-affecting runtime knobs."""
import json
import os
import pathlib
import subprocess

ROOT = pathlib.Path(__file__).resolve().parents[1]
BENCH = (ROOT / "scripts/bench-serial.sh").read_text(errors="replace")
cmd = "source env.sh >/dev/null 2>&1; env"
proc = subprocess.run(["bash", "-lc", cmd], cwd=ROOT, text=True, capture_output=True, check=True)
env = dict(line.split("=", 1) for line in proc.stdout.splitlines() if "=" in line)
prefixes = ("GGML_", "SYCL_", "ZE_", "ONEAPI_")
active = {k: v for k, v in sorted(env.items()) if k.startswith(prefixes)}
recorded = {k: v for k, v in active.items() if f'"{k}"' in BENCH}
unrecorded = {k: v for k, v in active.items() if k not in recorded}
report = {
    "audit": "runtime-environment-metrics-provenance",
    "active_prefixed_controls": active,
    "controls_recorded_by_serial_metrics": recorded,
    "active_controls_omitted_from_serial_metrics": unrecorded,
    "active_count": len(active),
    "recorded_count": len(recorded),
    "omitted_count": len(unrecorded),
}
out = ROOT / "benchmark/results/runtime-environment-metrics-provenance-audit-20260807.json"
out.parent.mkdir(parents=True, exist_ok=True)
out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
assert active
assert set(recorded) == {"GGML_SYCL_DISABLE_GRAPH", "ONEAPI_DEVICE_SELECTOR", "ZE_AFFINITY_MASK"}
assert {"GGML_SYCL_DISABLE_MUL_MAT_ADD_FUSE", "GGML_SYCL_DISABLE_MOE_DUAL_DOWN", "GGML_SYCL_DISABLE_MOE_DUAL_MULTITOKEN", "GGML_SYCL_DISABLE_QKV_SHARED_QUANT"} <= set(unrecorded)
print(out)
