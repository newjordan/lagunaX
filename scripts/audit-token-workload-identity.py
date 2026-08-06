#!/usr/bin/env python3
"""Audit whether Laguna throughput is bound to a reproducible token workload."""
import json
import re
from pathlib import Path

root = Path(__file__).resolve().parents[1]
harness = (root / "scripts/bench-serial.sh").read_text()
# The harness specifies only workload lengths; it supplies no prompt/token file,
# text, token IDs, RNG seed, or workload digest.
active_harness = "\n".join(line.split("#", 1)[0] for line in harness.splitlines())
controls = {
    "prompt_length": ' -p "$LX_PP"' in active_harness,
    "generation_length": ' -n "$LX_TG"' in active_harness,
    "prompt_text_or_file": bool(re.search(r"(?:^|\s)(?:--prompt|--file|prompt-file)(?:\s|=)", active_harness, re.MULTILINE)),
    "token_ids": "token_ids" in active_harness or "token-ids" in active_harness,
    "rng_seed": "--seed" in active_harness or " -s " in active_harness,
    "workload_digest": "workload_sha256" in active_harness or "prompt_sha256" in active_harness,
}
metrics_fields = []
for line in harness.splitlines():
    if '"window"' in line or '"pp512"' in line or '"tg128"' in line:
        metrics_fields.append(line.strip())
report = {
    "policy": "token_workload_identity",
    "canonical_harness": "scripts/bench-serial.sh",
    "controls": controls,
    "reproducible_token_workload_bound": all((controls["prompt_text_or_file"], controls["token_ids"], controls["rng_seed"], controls["workload_digest"])),
    "metrics_workload_lines": metrics_fields,
    "conclusion": "lengths are bound, token content and seed are not",
}
out = root / "results" / "token-workload-identity-audit-20260807.json"
out.write_text(json.dumps(report, indent=2) + "\n")
print(out)
