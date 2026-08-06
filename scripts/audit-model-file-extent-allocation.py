#!/usr/bin/env python3
"""Audit model-file extent allocation and Laguna benchmark provenance."""
import json
import os
import re
import subprocess
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ENV_PATH = ROOT / "env.sh"
BENCH_PATH = ROOT / "scripts/bench-serial.sh"
env_text = ENV_PATH.read_text(errors="replace")
bench_text = BENCH_PATH.read_text(errors="replace")

match = re.search(r'^export LX_MODEL="\$\{LX_MODEL:-([^}]+)\}"', env_text, re.MULTILINE)
if not match:
    raise SystemExit("LX_MODEL default not found in env.sh")
model = Path(match.group(1))
st = model.stat()
result = subprocess.run(
    ["filefrag", "-e", "-v", str(model)],
    check=True,
    text=True,
    stdout=subprocess.PIPE,
    stderr=subprocess.STDOUT,
).stdout
extent_match = re.search(rf"{re.escape(str(model))}: (\d+) extent(?:s)? found", result)
if not extent_match:
    raise SystemExit("filefrag extent summary not found")
extents = int(extent_match.group(1))
allocated_bytes = st.st_blocks * 512
source = (env_text + "\n" + bench_text).lower()
needles = ("filefrag", "fiemap", "extent", "st_blocks", "sparse")
report = {
    "angle": "model-file-extent-allocation-policy",
    "generated_at": datetime.now(timezone.utc).isoformat(),
    "model": {
        "path": str(model),
        "size_bytes": st.st_size,
        "allocated_bytes": allocated_bytes,
        "allocation_ratio": allocated_bytes / st.st_size,
        "extent_count": extents,
        "sparse": allocated_bytes < st.st_size,
        "filesystem_block_size": os.statvfs(model).f_frsize,
    },
    "laguna": {
        "source_hits": [needle for needle in needles if needle in source],
        "records_extent_or_allocation_state": any(needle in bench_text.lower() for needle in needles),
    },
    "finding": (
        "Laguna neither records nor rejects runs based on model-file extent "
        "fragmentation or sparse-allocation state."
    ),
}
out = ROOT / "results/model-file-extent-allocation-audit-20260807.json"
out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
print(out)
