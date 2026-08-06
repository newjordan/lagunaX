#!/usr/bin/env python3
"""Audit transparent-huge-page policy and Laguna benchmark provenance."""
import json
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ENV = (ROOT / "env.sh").read_text(errors="replace")
BENCH = (ROOT / "scripts/bench-serial.sh").read_text(errors="replace")

PATHS = {
    "enabled": "/sys/kernel/mm/transparent_hugepage/enabled",
    "defrag": "/sys/kernel/mm/transparent_hugepage/defrag",
    "shmem_enabled": "/sys/kernel/mm/transparent_hugepage/shmem_enabled",
    "khugepaged_defrag": "/sys/kernel/mm/transparent_hugepage/khugepaged/defrag",
    "khugepaged_pages_to_scan": "/sys/kernel/mm/transparent_hugepage/khugepaged/pages_to_scan",
    "khugepaged_scan_sleep_millisecs": "/sys/kernel/mm/transparent_hugepage/khugepaged/scan_sleep_millisecs",
}
TOKENS = ("transparent_hugepage", "khugepaged", "MADV_HUGEPAGE", "MADV_NOHUGEPAGE")


def read(path):
    try:
        return Path(path).read_text().strip()
    except OSError:
        return None


def selected(value):
    if value and "[" in value:
        return value.split("[", 1)[1].split("]", 1)[0]
    return value


raw = {name: read(path) for name, path in PATHS.items()}
report = {
    "angle": "transparent-huge-page-policy",
    "generated_at": datetime.now(timezone.utc).isoformat(),
    "live": {
        "raw": raw,
        "effective": {
            "enabled": selected(raw["enabled"]),
            "defrag": selected(raw["defrag"]),
            "shmem_enabled": selected(raw["shmem_enabled"]),
            "khugepaged_defrag": raw["khugepaged_defrag"],
            "khugepaged_pages_to_scan": raw["khugepaged_pages_to_scan"],
            "khugepaged_scan_sleep_millisecs": raw["khugepaged_scan_sleep_millisecs"],
        },
    },
    "laguna": {
        "env_policy_hits": [token for token in TOKENS if token in ENV],
        "bench_policy_hits": [token for token in TOKENS if token in BENCH],
        "metrics_record_policy": any(f'"{token}"' in BENCH for token in TOKENS),
    },
    "finding": (
        "The host uses madvise-only anonymous THP and defragmentation, disables "
        "shmem THP, and enables khugepaged defragmentation; Laguna neither "
        "configures nor records this effective policy."
    ),
}
out = ROOT / "results/transparent-huge-page-policy-audit-20260807.json"
out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
print(out)
