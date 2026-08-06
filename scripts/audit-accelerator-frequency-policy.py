#!/usr/bin/env python3
"""Audit accelerator frequency policy and benchmark provenance."""
import json
import re
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SOURCES = [ROOT / "env.sh", ROOT / "scripts/bench-serial.sh"]
FREQ_RE = re.compile(r"(?:min_freq|max_freq|boost_freq|cur_freq|act_freq|frequency|clock)", re.I)

freq_roots = sorted(Path("/sys/class/drm").glob("card*/device/tile*/gt*/freq0"))
devices = []
for root in freq_roots:
    values = {}
    for name in ("min_freq", "max_freq", "boost_freq", "cur_freq", "act_freq", "RP0_freq", "RPn_freq"):
        path = root / name
        if path.is_file():
            values[name] = int(path.read_text().strip())
    devices.append({"path": str(root), "values_mhz": values})

source_mentions = {}
for source in SOURCES:
    lines = source.read_text().splitlines()
    source_mentions[str(source.relative_to(ROOT))] = [
        {"line": i, "text": line.strip()} for i, line in enumerate(lines, 1)
        if FREQ_RE.search(line) and not line.lstrip().startswith("#")
    ]

result = {
    "generated_utc": datetime.now(timezone.utc).isoformat(),
    "angle": "accelerator-frequency-policy",
    "frequency_interfaces": devices,
    "active_source_mentions": source_mentions,
    "explicit_frequency_policy_active": any(source_mentions.values()),
    "summary": {
        "interfaces_found": len(devices),
        "configured_min_mhz": [d["values_mhz"].get("min_freq") for d in devices],
        "configured_max_mhz": [d["values_mhz"].get("max_freq") for d in devices],
    },
}
out = ROOT / "results" / "accelerator-frequency-policy-audit-20260807.json"
out.write_text(json.dumps(result, indent=2) + "\n")
print(out)
assert devices, "no accelerator frequency interfaces found"
assert not result["explicit_frequency_policy_active"], "active benchmark unexpectedly controls frequency"
