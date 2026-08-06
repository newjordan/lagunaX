#!/usr/bin/env python3
import json
from pathlib import Path

root = Path(__file__).resolve().parents[1]
env = (root / "env.sh").read_text(errors="replace")
bench = (root / "scripts/bench-serial.sh").read_text(errors="replace")
keys = ("GGML_SYCL_DISABLE_GRAPH", "GGML_SYCL_GRAPH")
counts = {key: {} for key in keys}
parsed = 0
for path in (root / "results").rglob("*.json"):
    if path.name == "sycl-graph-policy-audit-20260807.json":
        continue
    try:
        obj = json.loads(path.read_text(errors="replace"))
    except Exception:
        continue
    parsed += 1
    stack = [obj]
    while stack:
        value = stack.pop()
        if isinstance(value, dict):
            for key, child in value.items():
                if key in counts and child is not None:
                    label = str(child)
                    counts[key][label] = counts[key].get(label, 0) + 1
                stack.append(child)
        elif isinstance(value, list):
            stack.extend(value)
report = {
    "parsed_json_artifacts": parsed,
    "active_env_mentions": {key: key in env for key in keys},
    "serial_harness_mentions": {key: key in bench for key in keys},
    "recorded_values": counts,
    "active_default_disables_graph": 'GGML_SYCL_DISABLE_GRAPH:-1' in env,
    "effective_policy": "runtime auto-select" if 'GGML_SYCL_DISABLE_GRAPH:-1' in env else "unknown",
    "benchmark_invocations_covered": 2 if 'COMMON=(' in bench and '"${COMMON[@]}"' in bench else 0,
    "quality_result": "not measured by this policy-coverage audit",
}
assert report["active_default_disables_graph"]
assert report["effective_policy"] == "runtime auto-select"
assert report["benchmark_invocations_covered"] == 2
out = root / "results/sycl-graph-policy-audit-20260807.json"
out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
print(out)
