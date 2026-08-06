#!/usr/bin/env python3
"""Audit whether canonical Laguna runs capture accelerator utilization/bandwidth evidence."""
import json
import re
from pathlib import Path

root = Path(__file__).resolve().parents[1]
bench_path = root / "scripts/bench-serial.sh"
bench = bench_path.read_text()
patterns = {
    "xpu_smi": r"\bxpu-smi\b",
    "intel_gpu_top": r"\bintel_gpu_top\b",
    "sysfs_busy_percent": r"gt_busy_percent|engine_busy",
    "memory_bandwidth_counter": r"(?:memory|dram|vram)[_-]?(?:read|write)?[_-]?bandwidth|bytes[_-](?:read|written)",
}
controls = {name: bool(re.search(pattern, bench, re.I)) for name, pattern in patterns.items()}
json_paths = list((root / "results").rglob("*.json"))
telemetry_fields = 0
for path in json_paths:
    try:
        text = path.read_text(errors="replace")
    except OSError:
        continue
    if re.search(r'"(?:gpu_utilization|gt_busy_percent|memory_bandwidth|dram_bandwidth|vram_bandwidth)"\s*:', text, re.I):
        telemetry_fields += 1
report = {
    "audit": "accelerator-utilization-bandwidth-telemetry",
    "benchmark_source": str(bench_path.relative_to(root)),
    "active_capture_controls": controls,
    "active_capture_enabled": any(controls.values()),
    "result_json_artifacts_parsed": len(json_paths),
    "artifacts_with_utilization_or_bandwidth_fields": telemetry_fields,
    "conclusion": "Canonical serial runs do not capture accelerator occupancy or memory-bandwidth counters alongside throughput.",
}
print(json.dumps(report, indent=2))
