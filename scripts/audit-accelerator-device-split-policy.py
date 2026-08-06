#!/usr/bin/env python3
"""Audit accelerator device-selection and multi-device split policy."""
import json
import os
import re
import subprocess
from pathlib import Path

root = Path(__file__).resolve().parents[1]
env_text = (root / "env.sh").read_text()
bench_text = (root / "scripts/bench-serial.sh").read_text()
bin_path = os.environ.get("LX_LLAMA_BENCH", "/home/frosty40/turbo/worktrees/treebeard-base-control-latest/build-mmadd-decode/bin/llama-bench")
help_text = subprocess.run([bin_path, "--help"], text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, check=True).stdout

def default(flag: str):
    match = re.search(rf"^.*{re.escape(flag)}.*\(default: ([^)]+)\)", help_text, re.MULTILINE)
    return match.group(1) if match else None

active = env_text + "\n" + bench_text
report = {
    "audit": "accelerator-device-split-policy",
    "executable": bin_path,
    "executable_defaults": {
        "device": default("--device"),
        "split_mode": default("--split-mode"),
        "main_gpu": default("--main-gpu"),
        "tensor_split": default("--tensor-split"),
    },
    "active_policy": {
        "oneapi_device_selector": re.search(r'ONEAPI_DEVICE_SELECTOR[^\n]*:-([^}"]+)', env_text).group(1),
        "ze_affinity_mask": re.search(r'ZE_AFFINITY_MASK[^\n]*:-([^}"]+)', env_text).group(1),
        "harness_passes_device": bool(re.search(r'(^|\s)(-dev|--device)(\s|$)', bench_text)),
        "harness_passes_split_mode": bool(re.search(r'(^|\s)(-sm|--split-mode)(\s|$)', bench_text)),
        "harness_passes_main_gpu": bool(re.search(r'(^|\s)(-mg|--main-gpu)(\s|$)', bench_text)),
        "harness_passes_tensor_split": bool(re.search(r'(^|\s)(-ts|--tensor-split)(\s|$)', bench_text)),
    },
}
out = root / "results/accelerator-device-split-policy-audit-20260807.json"
out.write_text(json.dumps(report, indent=2) + "\n")
print(out)
