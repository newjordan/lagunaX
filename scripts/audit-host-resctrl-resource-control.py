#!/usr/bin/env python3
"""Audit Intel/AMD resctrl cache and memory-bandwidth controls and Laguna provenance."""
from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "host-resctrl-resource-control-audit-20260807.json"


def read(path: Path) -> str | None:
    try:
        return path.read_text().strip()
    except OSError:
        return None


mounts = read(Path("/proc/mounts")) or ""
resctrl_mounts = [line.split()[1] for line in mounts.splitlines() if len(line.split()) >= 3 and line.split()[2] == "resctrl"]
flags_text = read(Path("/proc/cpuinfo")) or ""
flags = sorted(set(re.findall(r"\b(?:cat_l3|cat_l2|mba|mba_MBps|cqm_llc|cqm_mbm_total|cqm_mbm_local)\b", flags_text)))
cmdline = read(Path("/proc/cmdline"))
resctrl = Path(resctrl_mounts[0]) if resctrl_mounts else Path("/sys/fs/resctrl")
info = {}
if (resctrl / "info").is_dir():
    for path in sorted((resctrl / "info").glob("*/*")):
        if path.is_file():
            info[str(path.relative_to(resctrl))] = read(path)

env_text = (ROOT / "env.sh").read_text()
harness_text = (ROOT / "scripts" / "bench-serial.sh").read_text()
source = (env_text + "\n" + harness_text).lower()
needles = ("resctrl", "schemata", "cat_l3", "cqm_mbm", "mba_sc", "rdtgroup")
report = {
    "generated_utc": datetime.now(timezone.utc).isoformat(),
    "scope": "resctrl cache allocation and memory-bandwidth control/monitoring",
    "live": {
        "cpu_resctrl_flags": flags,
        "kernel_cmdline": cmdline,
        "resctrl_mounts": resctrl_mounts,
        "resctrl_info_directory_exists": (resctrl / "info").is_dir(),
        "resctrl_info": info,
    },
    "laguna": {
        "canonical_environment": "env.sh",
        "serial_harness": "scripts/bench-serial.sh",
        "configures_resctrl": any(x in source for x in needles),
        "records_resctrl": any(x in harness_text.lower() for x in needles),
    },
}
OUT.write_text(json.dumps(report, indent=2) + "\n")
print(OUT)
