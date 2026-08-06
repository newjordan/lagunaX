#!/usr/bin/env python3
"""Audit Laguna KV-cache precision policy and metrics provenance."""
import json
import re
import subprocess
from pathlib import Path

root = Path(__file__).resolve().parents[1]
env_text = (root / "env.sh").read_text()
bench_text = (root / "scripts/bench-serial.sh").read_text()
probe = subprocess.run(
    ["bash", "-lc", f'source "{root / "env.sh"}"; "$LX_LLAMA_BENCH" --help'],
    text=True, capture_output=True, check=True,
)
help_text = probe.stdout + probe.stderr
k = re.search(r"--cache-type-k <t>\s+\(default:\s*([^)]+)\)", help_text)
v = re.search(r"--cache-type-v <t>\s+\(default:\s*([^)]+)\)", help_text)
if not (k and v):
    raise SystemExit("live llama-bench does not expose both KV-cache type controls")

def passed(flag: str) -> bool:
    return bool(re.search(rf"(?:^|[\s'\"]){re.escape(flag)}(?=[\s'\"])", bench_text))

artifact = {
    "live_defaults": {"cache_type_k": k.group(1).strip(), "cache_type_v": v.group(1).strip()},
    "canonical_values": {
        "cache_type_k": re.search(r'^export CTK="\$\{CTK:-([^}]+)\}"', env_text, re.M).group(1),
        "cache_type_v": re.search(r'^export CTV="\$\{CTV:-([^}]+)\}"', env_text, re.M).group(1),
    },
    "canonical_harness_passes_cache_type_k": passed("-ctk"),
    "canonical_harness_passes_cache_type_v": passed("-ctv"),
    "metrics_record_cache_type_k": '"ctk"' in bench_text,
    "metrics_record_cache_type_v": '"ctv"' in bench_text,
}
out = root / "results/kv-cache-precision-policy-audit-20260807.json"
out.write_text(json.dumps(artifact, indent=2) + "\n")
print(out)
