#!/usr/bin/env python3
"""Audit multi-generational LRU policy and Laguna benchmark provenance."""
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "benchmark/results/kernel-mglru-policy-audit-20260807.json"


def read(path: str) -> str | None:
    try:
        return Path(path).read_text().strip()
    except (FileNotFoundError, PermissionError):
        return None


enabled_raw = read("/sys/kernel/mm/lru_gen/enabled")
min_ttl_raw = read("/sys/kernel/mm/lru_gen/min_ttl_ms")
mask = int(enabled_raw, 0) if enabled_raw is not None else None
source = "\n".join(
    path.read_text(errors="replace")
    for path in (ROOT / "env.sh", ROOT / "scripts/bench-serial.sh")
)
controls = ("lru_gen", "min_ttl_ms", "mglru", "multi-gen lru")
artifact = {
    "live": {
        "interface_present": enabled_raw is not None,
        "enabled_raw": enabled_raw,
        "enabled_mask": mask,
        "enabled_features": {
            "main_switch": bool(mask is not None and mask & 0x1),
            "clear_refs_working_set": bool(mask is not None and mask & 0x2),
            "non_leaf_young_accessed": bool(mask is not None and mask & 0x4),
        },
        "minimum_generation_ttl_ms": int(min_ttl_raw) if min_ttl_raw is not None else None,
    },
    "laguna": {
        "env_or_harness_controls_policy": any(
            re.search(re.escape(item), source, re.I) for item in controls
        ),
        "serial_metrics_record_policy": any(
            re.search(re.escape(item), (ROOT / "scripts/bench-serial.sh").read_text(errors="replace"), re.I)
            for item in controls
        ),
    },
}
OUT.parent.mkdir(parents=True, exist_ok=True)
OUT.write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n")
print(OUT)
