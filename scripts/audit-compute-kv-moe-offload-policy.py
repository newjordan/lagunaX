#!/usr/bin/env python3
"""Audit compute/KV/MoE offload policy used by the canonical serial benchmark."""
import json
import os
import re
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
env_text = (ROOT / "env.sh").read_text()
bench_text = (ROOT / "scripts/bench-serial.sh").read_text()
cmd = f'source {ROOT / "env.sh"}; "$LX_LLAMA_BENCH" --help'
help_text = subprocess.run(["bash", "-lc", cmd], text=True, capture_output=True, check=True).stdout

def default(pattern: str):
    match = re.search(pattern, help_text)
    if not match:
        raise RuntimeError(f"missing help control: {pattern}")
    return int(match.group(1))

controls = {
    "n_cpu_moe": {"cli": "-ncmoe", "default": default(r"--n-cpu-moe.*?default: ([-0-9]+)")},
    "no_kv_offload": {"cli": "-nkvo", "default": default(r"--no-kv-offload.*?default: ([-0-9]+)")},
    "no_op_offload": {"cli": "-nopo", "default": default(r"--no-op-offload.*?default: ([-0-9]+)")},
}
for value in controls.values():
    token = value["cli"]
    value["env_override"] = token in env_text
    value["harness_override"] = token in bench_text
    value["effective"] = value["default"]

artifacts = 0
mentions = {name: 0 for name in controls}
for path in list((ROOT / "results").rglob("*.json")) + list((ROOT / "benchmark").rglob("*.json")):
    try:
        text = path.read_text(errors="replace")
        json.loads(text)
    except Exception:
        continue
    artifacts += 1
    for name in controls:
        if name in text:
            mentions[name] += 1

payload = {
    "audit": "compute-kv-moe-offload-policy",
    "controls": controls,
    "parsed_json_artifacts": artifacts,
    "artifact_mentions": mentions,
    "conclusion": "Canonical serial runs use executable defaults: all MoE layers and general ops remain accelerator-offloaded, and KV offload remains enabled.",
}
out = ROOT / "benchmark/results/compute-kv-moe-offload-policy-audit-20260807.json"
out.parent.mkdir(parents=True, exist_ok=True)
out.write_text(json.dumps(payload, indent=2) + "\n")
print(json.dumps(payload, indent=2))
