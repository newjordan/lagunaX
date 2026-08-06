#!/usr/bin/env python3
"""Audit active llama-bench multi-accelerator split controls."""
import json
import os
import re
import subprocess
from pathlib import Path

root = Path(__file__).resolve().parents[1]
env_text = (root / "env.sh").read_text()
bench_text = (root / "scripts/bench-serial.sh").read_text()
old = dict(os.environ)
cmd = f'source {root / "env.sh"} >/dev/null 2>&1; "$LX_LLAMA_BENCH" --help 2>&1'
help_text = subprocess.run(["bash", "-lc", cmd], check=True, text=True, capture_output=True, env=old).stdout

def default(pattern: str) -> str:
    m = re.search(pattern, help_text)
    if not m:
        raise SystemExit(f"missing help pattern: {pattern}")
    return m.group(1)

split_default = default(r"--split-mode <none\|layer\|row\|tensor>\s+\(default: ([^)]+)\)")
main_default = int(default(r"--main-gpu <i>\s+\(default: ([^)]+)\)"))
tensor_default = default(r"--tensor-split <ts0/ts1/\.\.>\s+\(default: ([^)]+)\)")
active = env_text + "\n" + bench_text
result = {
    "executable": os.path.realpath(subprocess.run(["bash", "-lc", f'source {root / "env.sh"} >/dev/null 2>&1; printf %s "$LX_LLAMA_BENCH"'], check=True, text=True, capture_output=True).stdout),
    "supported": {"split_modes": ["none", "layer", "row", "tensor"]},
    "defaults": {"split_mode": split_default, "main_gpu": main_default, "tensor_split": tensor_default},
    "active_overrides": {
        "split_mode": bool(re.search(r"(?:^|\s)(?:-sm|--split-mode)(?:\s|=)", active)),
        "main_gpu": bool(re.search(r"(?:^|\s)(?:-mg|--main-gpu)(?:\s|=)", active)),
        "tensor_split": bool(re.search(r"(?:^|\s)(?:-ts|--tensor-split)(?:\s|=)", active)),
    },
    "effective": {"split_mode": split_default, "main_gpu": main_default, "tensor_split": tensor_default},
    "device_constraints": {
        "oneapi_device_selector": re.search(r'export ONEAPI_DEVICE_SELECTOR="\$\{ONEAPI_DEVICE_SELECTOR:-([^}]+)\}"', env_text).group(1),
        "ze_affinity_mask": re.search(r'export ZE_AFFINITY_MASK="\$\{ZE_AFFINITY_MASK:-([^}]+)\}"', env_text).group(1),
    },
}
assert not any(result["active_overrides"].values())
assert result["effective"] == {"split_mode": "layer", "main_gpu": 0, "tensor_split": "0"}
out = root / "results" / "accelerator-tensor-split-policy-audit-20260807.json"
out.write_text(json.dumps(result, indent=2) + "\n")
print(out)
