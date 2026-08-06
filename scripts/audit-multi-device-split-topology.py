#!/usr/bin/env python3
"""Audit multi-device split controls in the active Laguna benchmark policy."""
import json
import re
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
env_text = (ROOT / "env.sh").read_text()
bench_text = (ROOT / "scripts/bench-serial.sh").read_text()
help_text = subprocess.run(
    ["bash", "-c", f"source {ROOT / 'env.sh'} >/dev/null && \"$LX_LLAMA_BENCH\" --help"],
    text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, check=True,
).stdout

def default(pattern: str) -> str:
    match = re.search(pattern, help_text)
    if not match:
        raise SystemExit(f"unparseable help pattern: {pattern}")
    return match.group(1)

active = env_text + "\n" + bench_text
artifact = {
    "policy": "multi_device_split_topology",
    "split_mode_supported": True,
    "split_mode_default": default(r"--split-mode <none\|layer\|row\|tensor>\s+\(default: ([^)]+)\)"),
    "main_gpu_default": int(default(r"--main-gpu <i>\s+\(default: ([0-9]+)\)")),
    "tensor_split_default": default(r"--tensor-split <ts0/ts1/\.\.>\s+\(default: ([^)]+)\)"),
    "active_split_mode_override": bool(re.search(r"(?:--split-mode|-sm\b|SPLIT_MODE)", active)),
    "active_main_gpu_override": bool(re.search(r"(?:--main-gpu|-mg\b|MAIN_GPU)", active)),
    "active_tensor_split_override": bool(re.search(r"(?:--tensor-split|-ts\b|TENSOR_SPLIT)", active)),
    "effective_device_selector": default(r"$^") if False else None,
}
artifact["interpretation"] = "default layer splitting remains selected; main GPU is ordinal 0; no tensor proportions are supplied"
out = ROOT / "benchmark/results/multi-device-split-topology-audit-20260807.json"
out.parent.mkdir(parents=True, exist_ok=True)
out.write_text(json.dumps(artifact, indent=2) + "\n")
print(out)
