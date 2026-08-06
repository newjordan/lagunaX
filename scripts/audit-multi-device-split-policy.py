#!/usr/bin/env python3
"""Audit Laguna's multi-device tensor-splitting policy."""
import json, os, re, subprocess
from pathlib import Path

root = Path(__file__).resolve().parents[1]
bench_text = (root / "scripts/bench-serial.sh").read_text()
env_text = (root / "env.sh").read_text()
binary = os.environ.get("LX_LLAMA_BENCH", str(root / "build/bin/llama-bench"))
help_text = subprocess.run(
    [binary, "--help"], text=True, stdout=subprocess.PIPE,
    stderr=subprocess.STDOUT, check=True,
).stdout

def default(pattern: str, name: str) -> str:
    match = re.search(pattern, help_text)
    if not match:
        raise SystemExit(f"could not derive {name} default")
    return match.group(1)

split_mode = default(r"--split-mode\s+<none\|layer\|row\|tensor>\s+\(default:\s*(\w+)\)", "split mode")
main_gpu = int(default(r"--main-gpu\s+<i>\s+\(default:\s*(\d+)\)", "main GPU"))
tensor_split = default(r"--tensor-split\s+<ts0/ts1/\.\.>\s+\(default:\s*([^\)]+)\)", "tensor split").strip()
active = bench_text + "\n" + env_text
source_override = bool(re.search(r"(?:--split-mode|--main-gpu|--tensor-split|-sm|-mg|-ts)(?:\s|=)", active))
process_override_keys = [
    key for key in ("SPLIT_MODE", "MAIN_GPU", "TENSOR_SPLIT") if key in os.environ
]
records = mentions = 0
for path in (root / "results").rglob("*.json"):
    try:
        obj = json.loads(path.read_text())
    except Exception:
        continue
    records += 1
    text = json.dumps(obj).lower()
    if any(term in text for term in ("split_mode", "split-mode", "tensor_split", "tensor-split", "main_gpu", "main-gpu")):
        mentions += 1
report = {
    "policy": "multi_device_tensor_split",
    "executable": binary,
    "supported_split_modes": ["none", "layer", "row", "tensor"],
    "default_split_mode": split_mode,
    "default_main_gpu": main_gpu,
    "default_tensor_split": tensor_split,
    "active_source_override": source_override,
    "process_environment_override_keys": process_override_keys,
    "historical_json_artifacts_parsed": records,
    "historical_artifacts_mentioning_split_controls": mentions,
}
out = root / "results" / "multi-device-split-policy-audit-20260807.json"
out.write_text(json.dumps(report, indent=2) + "\n")
print(out)
