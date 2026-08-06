#!/usr/bin/env python3
"""Audit Transparent Huge Pages policy and Laguna benchmark provenance."""
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "benchmark/results/transparent-hugepage-policy-audit-20260807.json"
THP = Path("/sys/kernel/mm/transparent_hugepage")


def read(path: Path) -> str | None:
    try:
        return path.read_text().strip()
    except (FileNotFoundError, PermissionError):
        return None


def selected(raw: str | None) -> str | None:
    if raw is None:
        return None
    match = re.search(r"\[([^]]+)\]", raw)
    return match.group(1) if match else None


enabled_raw = read(THP / "enabled")
defrag_raw = read(THP / "defrag")
shmem_raw = read(THP / "shmem_enabled")
khugepaged = THP / "khugepaged"
khugepaged_values = {
    name: read(khugepaged / name)
    for name in (
        "alloc_sleep_millisecs",
        "defrag",
        "full_scans",
        "max_ptes_none",
        "max_ptes_shared",
        "max_ptes_swap",
        "pages_collapsed",
        "pages_to_scan",
        "scan_sleep_millisecs",
    )
}
source_paths = (ROOT / "env.sh", ROOT / "scripts/bench-serial.sh")
source = "\n".join(path.read_text(errors="replace") for path in source_paths)
serial = source_paths[1].read_text(errors="replace")
controls = ("transparent_hugepage", "khugepaged", "MADV_HUGEPAGE", "THP")
artifact = {
    "live": {
        "interface_present": THP.is_dir(),
        "enabled_raw": enabled_raw,
        "enabled_selected": selected(enabled_raw),
        "defrag_raw": defrag_raw,
        "defrag_selected": selected(defrag_raw),
        "shmem_enabled_raw": shmem_raw,
        "shmem_enabled_selected": selected(shmem_raw),
        "khugepaged": khugepaged_values,
    },
    "laguna": {
        "env_or_harness_controls_policy": any(re.search(term, source, re.I) for term in controls),
        "serial_metrics_record_policy": any(re.search(term, serial, re.I) for term in controls),
    },
}
OUT.parent.mkdir(parents=True, exist_ok=True)
OUT.write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n")
print(OUT)
