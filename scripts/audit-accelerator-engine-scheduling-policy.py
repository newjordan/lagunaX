#!/usr/bin/env python3
"""Audit accelerator engine timeout policy and benchmark provenance."""
import json
from pathlib import Path

root = Path(__file__).resolve().parents[1]
device = Path("/sys/class/drm/renderD128/device").resolve()


def read_int(path: Path):
    try:
        return int(path.read_text().strip())
    except (OSError, ValueError):
        return None


engine_controls = []
for path in sorted(device.glob("tile*/gt*/engines/*/job_timeout_ms")):
    engine_controls.append(
        {
            "engine": path.parent.name,
            "gt": path.parents[2].name,
            "path": str(path),
            "job_timeout_ms": read_int(path),
        }
    )

sources = {
    name: (root / name).read_text()
    for name in ("env.sh", "scripts/bench-serial.sh")
}
needles = ("job_timeout_ms", "preempt_timeout_ms", "timeslice_duration_ms")
artifact = {
    "device": str(device),
    "engine_controls": engine_controls,
    "laguna_source_mentions": {
        name: {needle: needle in text for needle in needles}
        for name, text in sources.items()
    },
    "conclusion": (
        "Kernel-exported accelerator engine timeout policy is readable, but "
        "Laguna's canonical serial path neither controls nor records it."
    ),
}
out = root / "results/accelerator-engine-scheduling-policy-audit-20260807.json"
out.write_text(json.dumps(artifact, indent=2) + "\n")
assert engine_controls
assert all(item["job_timeout_ms"] is not None for item in engine_controls)
assert all(
    not found
    for source in artifact["laguna_source_mentions"].values()
    for found in source.values()
)
print(out)
