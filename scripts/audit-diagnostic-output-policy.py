#!/usr/bin/env python3
"""Audit benchmark diagnostic-output policy and historical coverage."""
import json
import os
from pathlib import Path

root = Path(__file__).resolve().parents[1]
harness = (root / "scripts/bench-serial.sh").read_text()
env_source = (root / "env.sh").read_text()
bench = os.environ.get("LX_LLAMA_BENCH", "/home/frosty40/turbo/worktrees/treebeard-base-control-latest/build-mmadd-decode/bin/llama-bench")

controls = {
    "verbose": ("--verbose", " -v "),
    "progress": ("--progress",),
    "sycl_pi_trace": ("SYCL_PI_TRACE",),
    "level_zero_tracing": ("ZE_ENABLE_TRACING_LAYER",),
}
active = {
    name: any(token in harness or token in env_source or token in os.environ for token in tokens)
    for name, tokens in controls.items()
}
mentions = {name: 0 for name in controls}
parsed = 0
out = root / "benchmark/results/diagnostic-output-policy-audit-20260807.json"
for path in list((root / "results").rglob("*.json")) + list((root / "benchmark/results").rglob("*.json")):
    if path == out:
        continue
    try:
        text = path.read_text()
        json.loads(text)
        parsed += 1
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        continue
    for name, tokens in controls.items():
        mentions[name] += sum(text.count(token) for token in tokens)

result = {
    "angle": "benchmark diagnostic output and accelerator tracing overhead",
    "executable": bench,
    "executable_controls_observed": {"verbose": "-v, --verbose", "progress": "--progress"},
    "active_policy": active,
    "parsed_json_artifacts": parsed,
    "historical_mentions": mentions,
}
out = root / "benchmark/results/diagnostic-output-policy-audit-20260807.json"
out.parent.mkdir(parents=True, exist_ok=True)
out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
print(out)
