#!/usr/bin/env python3
"""Audit whether Laguna forbids host tensor buffers via llama-bench --no-host."""
import json
import os
import re
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
env_text = (ROOT / "env.sh").read_text()
bench_text = (ROOT / "scripts/bench-serial.sh").read_text()
cmd = f"source {ROOT / 'env.sh'} >/dev/null && printf '%s' \"$LX_LLAMA_BENCH\""
bench = subprocess.check_output(["bash", "-c", cmd], text=True)
help_text = subprocess.run(
    ["bash", "-c", f"source {ROOT / 'env.sh'} >/dev/null && \"$LX_LLAMA_BENCH\" --help"],
    text=True,
    stdout=subprocess.PIPE,
    stderr=subprocess.STDOUT,
    check=False,
).stdout
match = re.search(r"--no-host <0\|1>\s+\(default: ([01])\)", help_text)
if not match:
    raise SystemExit("active llama-bench does not expose a parseable --no-host default")
source_override = bool(re.search(r"(?:--no-host|NO_HOST)", env_text + "\n" + bench_text))
env_override = "NO_HOST" in os.environ
artifact = {
    "policy": "host_tensor_buffers",
    "executable": bench,
    "no_host_supported": True,
    "executable_default_no_host": int(match.group(1)),
    "active_source_override": source_override,
    "process_environment_override": env_override,
    "effective_no_host": int(os.environ.get("NO_HOST", match.group(1))),
    "interpretation": "host tensor buffers remain permitted" if int(os.environ.get("NO_HOST", match.group(1))) == 0 else "host tensor buffers forbidden",
}
out = ROOT / "benchmark/results/host-tensor-buffer-policy-audit-20260807.json"
out.parent.mkdir(parents=True, exist_ok=True)
out.write_text(json.dumps(artifact, indent=2) + "\n")
print(out)
