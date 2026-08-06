#!/usr/bin/env python3
"""Read-only audit of host virtualization and CPU-steal-time provenance."""
import json
import re
import subprocess
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / f"host-virtualization-steal-time-audit-{date.today():%Y%m%d}.json"

cpuinfo = Path("/proc/cpuinfo").read_text()
stat_lines = Path("/proc/stat").read_text().splitlines()
env = (ROOT / "env.sh").read_text()
harness = (ROOT / "scripts/bench-serial.sh").read_text()

cpu_rows = []
for line in stat_lines:
    fields = line.split()
    if not re.fullmatch(r"cpu\d+", fields[0] if fields else ""):
        continue
    cpu_rows.append({"cpu": fields[0], "steal_ticks": int(fields[8]) if len(fields) > 8 else None})

try:
    detected_virtualization = subprocess.run(
        ["systemd-detect-virt"], capture_output=True, text=True, check=False
    ).stdout.strip() or "none"
except FileNotFoundError:
    detected_virtualization = "unavailable"

combined = env + "\n" + harness
artifact = {
    "audit": "host-virtualization-steal-time-policy",
    "detected_virtualization": detected_virtualization,
    "cpuinfo_hypervisor_flag": bool(re.search(r"^flags\s*:.*\bhypervisor\b", cpuinfo, re.M)),
    "logical_cpu_count": len(cpu_rows),
    "per_cpu_steal_ticks": cpu_rows,
    "total_steal_ticks": sum(row["steal_ticks"] or 0 for row in cpu_rows),
    "laguna_rejects_virtualized_hosts": bool(re.search(r"detect-virt|hypervisor.*reject|reject.*virtual", combined, re.I)),
    "laguna_records_virtualization": bool(re.search(r"virtualization|detect-virt|hypervisor_flag", harness, re.I)),
    "laguna_records_cpu_steal": bool(re.search(r"steal[_ -]?(time|ticks)|cpu_steal", harness, re.I)),
}
OUT.parent.mkdir(parents=True, exist_ok=True)
OUT.write_text(json.dumps(artifact, indent=2) + "\n")
print(OUT.relative_to(ROOT))
