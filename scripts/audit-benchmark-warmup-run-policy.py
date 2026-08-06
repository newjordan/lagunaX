#!/usr/bin/env python3
"""Audit warmup-run policy and provenance for Laguna's canonical benchmark."""
import json
import re
import subprocess
from pathlib import Path

root = Path(__file__).resolve().parents[1]
env_text = (root / "env.sh").read_text()
bench_text = (root / "scripts/bench-serial.sh").read_text()
help_text = subprocess.run(
    ["bash", "-lc", f'source "{root}/env.sh" && "$LX_LLAMA_BENCH" --help 2>&1'],
    text=True,
    capture_output=True,
    check=True,
).stdout

supported = "--no-warmup" in help_text
source = env_text + "\n" + bench_text
canonical_disables_warmup = bool(re.search(r"(^|\s)--no-warmup(?:\s|$)", source))
metrics_record_warmup = bool(re.search(r'["\x27](?:warmup|warmup_enabled|no_warmup)["\x27]\s*:', bench_text))
invocations = len(re.findall(r'\$LX_LLAMA_BENCH"\s+"\$\{COMMON\[@\]\}"', bench_text))

payload = {
    "angle": "benchmark_warmup_run_policy",
    "live_help": {
        "no_warmup_supported": supported,
        "default_warmup_enabled": supported,
    },
    "canonical_sources": {
        "disables_warmup": canonical_disables_warmup,
        "metrics_record_warmup_policy": metrics_record_warmup,
        "llama_bench_invocations": invocations,
        "effective_warmup_sequences_per_serial_run": invocations if supported and not canonical_disables_warmup else 0,
    },
}
assert supported
assert not canonical_disables_warmup
assert not metrics_record_warmup
assert invocations == 2
out = root / "benchmark/results/benchmark-warmup-run-policy-audit-20260807.json"
out.parent.mkdir(parents=True, exist_ok=True)
out.write_text(json.dumps(payload, indent=2) + "\n")
print(out)
