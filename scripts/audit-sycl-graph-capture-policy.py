#!/usr/bin/env python3
"""Audit SYCL graph-capture policy and historical result provenance."""
import json
import os
import pathlib
import re
import subprocess

root = pathlib.Path(__file__).resolve().parents[1]
env_path = root / "env.sh"
bench_path = root / "scripts" / "bench-serial.sh"
env_text = env_path.read_text()
bench_text = bench_path.read_text()

match = re.search(
    r'export GGML_SYCL_DISABLE_GRAPH="\$\{GGML_SYCL_DISABLE_GRAPH:-([01])\}"',
    env_text,
)
if not match:
    raise RuntimeError("could not resolve GGML_SYCL_DISABLE_GRAPH default")

resolved = subprocess.check_output(
    ["bash", "-c", f"source {env_path} >/dev/null 2>&1; printf %s \"$GGML_SYCL_DISABLE_GRAPH\""],
    text=True,
)
parsed = 0
mentions = {"0": 0, "1": 0}
for path in (root / "results").rglob("*.json"):
    try:
        obj = json.loads(path.read_text())
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        continue
    parsed += 1
    stack = [obj]
    while stack:
        item = stack.pop()
        if isinstance(item, dict):
            for key, value in item.items():
                if key == "GGML_SYCL_DISABLE_GRAPH" and str(value) in mentions:
                    mentions[str(value)] += 1
                stack.append(value)
        elif isinstance(item, list):
            stack.extend(item)

artifact = {
    "audit": "sycl-graph-capture-policy",
    "source_default": int(match.group(1)),
    "effective_value": int(resolved),
    "graph_capture_enabled": resolved == "0",
    "serial_harness_passes_policy_to_cli": bool(re.search(r'(^|\s)(?:--?[A-Za-z0-9-]*graph[A-Za-z0-9-]*)\s+', bench_text, re.MULTILINE)),
    "serial_metrics_record_policy": '"GGML_SYCL_DISABLE_GRAPH": os.environ.get("GGML_SYCL_DISABLE_GRAPH")' in bench_text,
    "historical_json_artifacts_parsed": parsed,
    "historical_policy_mentions": mentions,
}
assert artifact["source_default"] == 1
assert artifact["effective_value"] == 1
assert not artifact["graph_capture_enabled"]
assert not artifact["serial_harness_passes_policy_to_cli"]
assert artifact["serial_metrics_record_policy"]
out = root / "results" / "sycl-graph-capture-policy-audit-20260807.json"
out.write_text(json.dumps(artifact, indent=2) + "\n")
print(out)
