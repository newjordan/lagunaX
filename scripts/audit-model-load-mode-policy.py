#!/usr/bin/env python3
"""Audit model load-mode configuration and provenance in canonical Laguna benchmarks."""
import json
import re
import subprocess
from pathlib import Path

root = Path(__file__).resolve().parents[1]
env_text = (root / "env.sh").read_text()
bench_text = (root / "scripts/bench-serial.sh").read_text()
source = env_text + "\n" + bench_text

probe = subprocess.run(
    ["bash", "-lc", 'source ./env.sh; "$LX_LLAMA_BENCH" --help 2>&1'],
    cwd=root, text=True, capture_output=True, check=True,
).stdout
load = re.search(r"(?:-lm, )?--load-mode <([^>]+)>.*?\(default: ([^)]+)\)", probe)
mmap = re.search(r"(?:-mmp, )?--mmap <([^>]+)>.*?DEPRECATED IN FAVOUR OF --load-mode", probe)
assert load and mmap, "llama-bench load-mode contract not found"
accepted = load.group(1).split("|")

artifact = {
    "audit": "model-load-mode-policy",
    "executable_contract": {
        "accepted_load_modes": accepted,
        "default_load_mode": load.group(2),
        "legacy_mmap_accepted": mmap.group(1).split("|"),
        "legacy_mmap_deprecated": True,
    },
    "canonical_configured": bool(re.search(r"(?:--load-mode|-lm\b|--mmap|-mmp\b|\bLOAD_MODE=)", source)),
    "metrics_recorded": bool(re.search(r'["\'](?:load_mode|mmap|mlock|direct_io)["\']\s*:', bench_text)),
}
out = root / "results/model-load-mode-policy-audit-20260807.json"
out.write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n")
print(out)
