#!/usr/bin/env python3
"""Audit host VM swap and metadata-cache reclaim policy provenance."""
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "benchmark/results/host-vm-swap-cache-reclaim-audit-20260807.json"
CONTROLS = {
    "swappiness": Path("/proc/sys/vm/swappiness"),
    "page_cluster": Path("/proc/sys/vm/page-cluster"),
    "vfs_cache_pressure": Path("/proc/sys/vm/vfs_cache_pressure"),
    "watermark_scale_factor": Path("/proc/sys/vm/watermark_scale_factor"),
}


def read_meminfo() -> dict[str, int]:
    values = {}
    for line in Path("/proc/meminfo").read_text().splitlines():
        key, value = line.split(":", 1)
        match = re.search(r"\d+", value)
        if match:
            values[key] = int(match.group())
    return values


live = {name: int(path.read_text().strip()) for name, path in CONTROLS.items()}
meminfo = read_meminfo()
source_paths = (ROOT / "env.sh", ROOT / "scripts/bench-serial.sh")
source = "\n".join(path.read_text(errors="replace") for path in source_paths)
metrics = source_paths[1].read_text(errors="replace")
names = tuple(CONTROLS)
artifact = {
    "live": live,
    "memory_kib": {
        "swap_total": meminfo.get("SwapTotal", 0),
        "swap_free": meminfo.get("SwapFree", 0),
        "swap_cached": meminfo.get("SwapCached", 0),
        "cached": meminfo.get("Cached", 0),
        "s_reclaimable": meminfo.get("SReclaimable", 0),
    },
    "derived": {
        "swap_configured": meminfo.get("SwapTotal", 0) > 0,
        "swap_used_kib": meminfo.get("SwapTotal", 0) - meminfo.get("SwapFree", 0),
    },
    "laguna": {
        "env_or_harness_controls_policy": any(
            re.search(rf"\b{re.escape(name)}\b", source) for name in names
        ),
        "serial_metrics_record_policy": any(name in metrics for name in names),
        "serial_metrics_record_swap_state": any(
            name in metrics for name in ("SwapTotal", "SwapFree", "SwapCached")
        ),
    },
}
OUT.parent.mkdir(parents=True, exist_ok=True)
OUT.write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n")
print(OUT)
