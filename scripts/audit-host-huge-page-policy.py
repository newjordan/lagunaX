#!/usr/bin/env python3
"""Audit transparent/explicit huge-page policy and provenance for Laguna."""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "host-huge-page-policy-audit-20260807.json"
ENV = ROOT / "env.sh"
HARNESS = ROOT / "scripts" / "bench-serial.sh"


def read(path: str) -> str | None:
    try:
        return Path(path).read_text().strip()
    except OSError:
        return None


def meminfo() -> dict[str, str]:
    wanted = {
        "AnonHugePages", "ShmemHugePages", "FileHugePages",
        "HugePages_Total", "HugePages_Free", "Hugepagesize",
    }
    values: dict[str, str] = {}
    for line in Path("/proc/meminfo").read_text().splitlines():
        key, _, value = line.partition(":")
        if key in wanted:
            values[key] = value.strip()
    return values


def main() -> None:
    env_text = ENV.read_text()
    harness_text = HARNESS.read_text()
    source = (env_text + "\n" + harness_text).lower()
    controls = ("transparent_hugepage", "hugetlb", "hugepages", "madvise")
    report = {
        "scope": "host huge-page policy and benchmark provenance",
        "live": {
            "transparent_hugepage_enabled": read("/sys/kernel/mm/transparent_hugepage/enabled"),
            "transparent_hugepage_defrag": read("/sys/kernel/mm/transparent_hugepage/defrag"),
            "transparent_hugepage_shmem_enabled": read("/sys/kernel/mm/transparent_hugepage/shmem_enabled"),
            "meminfo": meminfo(),
        },
        "laguna": {
            "canonical_environment": str(ENV.relative_to(ROOT)),
            "serial_harness": str(HARNESS.relative_to(ROOT)),
            "configures_huge_page_policy": any(token in source for token in controls),
            "records_huge_page_state": any(
                token in harness_text.lower() for token in controls
            ),
        },
    }
    OUT.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(OUT.relative_to(ROOT))


if __name__ == "__main__":
    main()
