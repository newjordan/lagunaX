#!/usr/bin/env python3
"""Audit compiler/build optimization provenance for the active Laguna binary."""
import json
import os
from pathlib import Path

root = Path(__file__).resolve().parents[1]
active = {
    "env.sh": (root / "env.sh").read_text(),
    "scripts/bench-serial.sh": (root / "scripts/bench-serial.sh").read_text(),
}
controls = ["CMAKE_BUILD_TYPE", "CMAKE_C_FLAGS", "CMAKE_CXX_FLAGS", "-march=", "-mtune=", "-flto", "IPO"]
source_mentions = {
    name: [control for control in controls if control in text]
    for name, text in active.items()
}
bin_path = Path(os.environ.get("LX_LLAMA_BENCH", root / "missing-llama-bench"))
report = {
    "direction": "compiler optimization and binary build provenance",
    "active_source_mentions": source_mentions,
    "active_source_explicit_control_count": sum(map(len, source_mentions.values())),
    "process_environment": {key: os.environ[key] for key in controls if key in os.environ},
    "llama_bench_path": str(bin_path),
    "llama_bench_exists": bin_path.exists(),
    "adjacent_cmake_cache_exists": (bin_path.parent.parent / "CMakeCache.txt").exists(),
    "quality_result": "not measured by this provenance audit",
}
out = root / "results/compiler-optimization-provenance-audit-20260807.json"
out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
print(out)
