#!/usr/bin/env python3
"""Audit VM dirty-page thresholds and writeback cadence provenance."""
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "benchmark/results/host-vm-dirty-writeback-policy-audit-20260807.json"

SYSCTLS = {
    "dirty_background_bytes": "/proc/sys/vm/dirty_background_bytes",
    "dirty_background_ratio": "/proc/sys/vm/dirty_background_ratio",
    "dirty_bytes": "/proc/sys/vm/dirty_bytes",
    "dirty_expire_centisecs": "/proc/sys/vm/dirty_expire_centisecs",
    "dirty_ratio": "/proc/sys/vm/dirty_ratio",
    "dirty_writeback_centisecs": "/proc/sys/vm/dirty_writeback_centisecs",
}


def read_int(path: str) -> int:
    return int(Path(path).read_text().strip())


def meminfo() -> dict[str, int]:
    values = {}
    for line in Path("/proc/meminfo").read_text().splitlines():
        key, raw = line.split(":", 1)
        if key in {"Dirty", "Writeback", "WritebackTmp"}:
            values[key] = int(raw.strip().split()[0])
    return values


policy = {name: read_int(path) for name, path in SYSCTLS.items()}
source = "\n".join(
    path.read_text(errors="replace")
    for path in (ROOT / "env.sh", ROOT / "scripts/bench-serial.sh")
)
metrics = (ROOT / "scripts/bench-serial.sh").read_text(errors="replace")
terms = tuple(SYSCTLS) + ("WritebackTmp",)
artifact = {
    "live": {
        "policy": policy,
        "memory_kib": meminfo(),
        "effective_threshold_mode": {
            "background": "bytes" if policy["dirty_background_bytes"] else "ratio",
            "foreground": "bytes" if policy["dirty_bytes"] else "ratio",
        },
    },
    "laguna": {
        "env_or_harness_controls_policy": any(
            re.search(rf"\b{re.escape(term)}\b", source) for term in SYSCTLS
        ),
        "serial_metrics_record_policy_or_dirty_state": any(
            re.search(rf"\b{re.escape(term)}\b", metrics) for term in terms
        ),
    },
}
OUT.parent.mkdir(parents=True, exist_ok=True)
OUT.write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n")
print(OUT)
