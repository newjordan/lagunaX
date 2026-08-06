#!/usr/bin/env python3
"""Audit Laguna's model-layer accelerator-offload policy."""
import json
import os
import re
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
env_text = (ROOT / "env.sh").read_text()
bench_text = (ROOT / "scripts/bench-serial.sh").read_text()
bench = os.environ.get("LX_LLAMA_BENCH")
if not bench:
    match = re.search(r'export LX_BIN="\$\{LX_BIN:-([^}]+)\}"', env_text)
    if not match:
        raise SystemExit("cannot resolve LX_BIN")
    bench = str(Path(match.group(1)) / "llama-bench")
help_text = subprocess.run([bench, "--help"], text=True, capture_output=True, check=True).stdout
if not re.search(r"(?:-ngl|--n-gpu-layers)\s+<n>", help_text):
    raise SystemExit("GPU-layer offload control absent from executable help")
env_match = re.search(r'export NGL="\$\{NGL:-(\d+)\}"', env_text)
if not env_match:
    raise SystemExit("NGL default absent from env.sh")
common_pass = bool(re.search(r'-ngl\s+"\$NGL"', bench_text))
values = []
parsed = 0
for path in ROOT.rglob("*.json"):
    try:
        data = json.loads(path.read_text())
        parsed += 1
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        continue
    stack = [data]
    while stack:
        item = stack.pop()
        if isinstance(item, dict):
            for key, value in item.items():
                if key in {"n_gpu_layers", "gpu_layers", "n-gpu-layers"} and isinstance(value, int):
                    values.append(value)
                stack.append(value)
        elif isinstance(item, list):
            stack.extend(item)
payload = {
    "executable": bench,
    "executable_supports_gpu_layer_offload": True,
    "configured_ngl": int(env_match.group(1)),
    "serial_harness_passes_ngl": common_pass,
    "effective_policy": "request up to 99 model layers on accelerator" if common_pass else "not propagated",
    "parsed_json_artifacts": parsed,
    "historical_recorded_values": sorted(set(values)),
    "historical_value_count": len(values),
}
out = ROOT / "results" / "model-layer-offload-policy-audit-20260807.json"
out.write_text(json.dumps(payload, indent=2) + "\n")
print(out)
print(json.dumps(payload, indent=2))
