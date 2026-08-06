#!/usr/bin/env python3
"""Audit whether the recorded context-size policy reaches llama-bench."""
import json
import re
from pathlib import Path

root = Path(__file__).resolve().parents[1]
env = (root / "env.sh").read_text()
harness_path = root / "scripts/bench-serial.sh"
harness = harness_path.read_text()

ctx_default = int(re.search(r'export CTX="\$\{CTX:-(\d+)\}"', env).group(1))
common = re.search(r"COMMON=\(\n(.*?)\n\)", harness, re.S).group(1)
invocation_uses_ctx = bool(re.search(r'(^|\s)(-c|--ctx-size)\s+"?\$CTX', common))
metrics_records_ctx = bool(re.search(r'"ctx": int\("\$CTX"\)', harness))

artifact = {
    "policy": "context-size-propagation",
    "env_ctx_default": ctx_default,
    "llama_bench_common_uses_ctx": invocation_uses_ctx,
    "metrics_records_ctx": metrics_records_ctx,
    "recorded_ctx_is_not_an_effective_benchmark_argument": metrics_records_ctx and not invocation_uses_ctx,
    "sources": ["env.sh", "scripts/bench-serial.sh"],
}
out = root / "benchmark/results/context-size-propagation-audit-20260807.json"
out.parent.mkdir(parents=True, exist_ok=True)
out.write_text(json.dumps(artifact, indent=2) + "\n")
print(out)

assert ctx_default == 8192
assert not invocation_uses_ctx
assert metrics_records_ctx
assert artifact["recorded_ctx_is_not_an_effective_benchmark_argument"]
