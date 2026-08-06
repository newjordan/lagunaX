#!/usr/bin/env python3
"""Audit model backing filesystem/block-device I/O policy and provenance."""
import json, os
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ENV = (ROOT / "env.sh").read_text(errors="replace")
BENCH = (ROOT / "scripts/bench-serial.sh").read_text(errors="replace")
model = os.environ.get("LX_MODEL", "/mnt/data2tb/laguna/archived/Laguna-XS-2.1-GGUF-q4-20260705/Laguna-XS-2.1-Q4_K_M.gguf")

def read(path):
    try: return Path(path).read_text().strip()
    except OSError: return None

def mount_for(path):
    best = None
    for line in (read("/proc/self/mountinfo") or "").splitlines():
        left, sep, right = line.partition(" - ")
        if not sep: continue
        fields, post = left.split(), right.split()
        mountpoint = fields[4].replace("\\040", " ")
        if path == mountpoint or path.startswith(mountpoint.rstrip("/") + "/"):
            if best is None or len(mountpoint) > len(best[0]): best = (mountpoint, fields, post)
    return best

mount = mount_for(model)
info = {}
if mount:
    mp, fields, post = mount
    major_minor = fields[2]
    dev = Path("/sys/dev/block") / major_minor
    resolved = dev.resolve() if dev.exists() else None
    disk = resolved
    while disk and not (disk / "queue").is_dir() and disk != disk.parent: disk = disk.parent
    queue = disk / "queue" if disk else None
    info = {"mountpoint": mp, "major_minor": major_minor, "fstype": post[0], "source": post[1],
            "mount_options": fields[5].split(","), "super_options": post[2].split(",") if len(post)>2 else [],
            "block_device": disk.name if disk else None,
            "queue": {k: read(queue/k) if queue else None for k in ("scheduler","read_ahead_kb","nr_requests","rotational","nomerges","rq_affinity")}}
tokens = ("read_ahead_kb", "nr_requests", "nomerges", "rq_affinity", "block scheduler", "mount_options")
report = {"angle":"model-storage-io-policy", "generated_at":datetime.now(timezone.utc).isoformat(),
          "model":model, "model_exists":Path(model).exists(), "storage":info,
          "laguna":{"env_policy_hits":[t for t in tokens if t in ENV], "bench_policy_hits":[t for t in tokens if t in BENCH],
                    "metrics_record_policy":any(f'\"{t}\"' in BENCH for t in tokens)}}
out = ROOT / "results/model-storage-io-policy-audit-20260807.json"
out.write_text(json.dumps(report, indent=2, sort_keys=True)+"\n")
print(out)
