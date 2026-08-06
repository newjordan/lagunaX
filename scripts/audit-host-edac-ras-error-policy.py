#!/usr/bin/env python3
"""Audit host EDAC/RAS error counters and Laguna run-validity provenance."""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def read(path: Path):
    try:
        return path.read_text().strip()
    except OSError:
        return None


edac = []
for mc in sorted(Path("/sys/devices/system/edac/mc").glob("mc[0-9]*")):
    entry = {"controller": mc.name}
    for name in ("ce_count", "ue_count", "seconds_since_reset", "size_mb"):
        entry[name] = read(mc / name)
    dimms = []
    for dimm in sorted(mc.glob("dimm[0-9]*")):
        dimms.append({
            "dimm": dimm.name,
            "label": read(dimm / "dimm_label"),
            "ce_count": read(dimm / "dimm_ce_count"),
            "ue_count": read(dimm / "dimm_ue_count"),
        })
    entry["dimms"] = dimms
    edac.append(entry)

mce_banks = []
for bank in sorted(Path("/sys/devices/system/machinecheck").glob("machinecheck*/bank*")):
    mce_banks.append({"path": str(bank), "ctl": read(bank / "ctl") if bank.is_dir() else read(bank)})

sources = {p: (ROOT / p).read_text() for p in ("env.sh", "scripts/bench-serial.sh")}
needles = ("ce_count", "ue_count", "EDAC", "machinecheck", "rasdaemon", "mcelog")
mentions = {
    source: {needle: needle.lower() in text.lower() for needle in needles}
    for source, text in sources.items()
}
artifact = {
    "edac_controllers": edac,
    "machine_check_banks": mce_banks,
    "laguna_source_mentions": mentions,
    "conclusion": (
        "Laguna's canonical serial path neither records EDAC/MCE state nor rejects runs "
        "when corrected or uncorrected hardware-error counters change."
    ),
}
out = ROOT / "results/host-edac-ras-error-policy-audit-20260807.json"
out.write_text(json.dumps(artifact, indent=2) + "\n")
assert not any(found for source in mentions.values() for found in source.values())
assert Path("/sys/devices/system/edac").exists() or Path("/sys/devices/system/machinecheck").exists()
print(out)
