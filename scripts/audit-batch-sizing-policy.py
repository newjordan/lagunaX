#!/usr/bin/env python3
"""Audit active logical/physical batch sizing without running inference."""
import json
import os
import re
import subprocess
from datetime import datetime, timezone
from pathlib import Path

root = Path(__file__).resolve().parents[1]
env_text = (root / "env.sh").read_text()
bench_text = (root / "scripts/bench-serial.sh").read_text()
binary = os.environ.get("LX_LLAMA_BENCH", "/home/frosty40/turbo/worktrees/treebeard-base-control-latest/build-mmadd-decode/bin/llama-bench")
help_text = subprocess.run([binary, "--help"], text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, check=True).stdout

def default(flag: str) -> int:
    match = re.search(rf"{re.escape(flag)}[^\n]*\(default: (\d+)\)", help_text)
    if not match:
        raise SystemExit(f"missing default for {flag}")
    return int(match.group(1))

def env_default(name: str) -> int:
    match = re.search(rf'export {name}="\$\{{{name}:-([0-9]+)\}}"', env_text)
    if not match:
        raise SystemExit(f"missing env default for {name}")
    return int(match.group(1))

bbatch = env_default("BBATCH")
ubatch = env_default("UBATCH")
if not re.search(r'-b "\$BBATCH"', bench_text) or not re.search(r'-ub "\$UBATCH"', bench_text):
    raise SystemExit("serial harness does not propagate both batch controls")
payload = {
    "generated_at": datetime.now(timezone.utc).isoformat(),
    "binary": binary,
    "executable_defaults": {"batch_size": default("--batch-size"), "ubatch_size": default("--ubatch-size")},
    "active_policy": {"batch_size": bbatch, "ubatch_size": ubatch},
    "overrides_executable_ubatch_default": ubatch != default("--ubatch-size"),
    "ubatch_multiplier_vs_default": ubatch / default("--ubatch-size"),
    "logical_to_physical_batch_ratio": bbatch / ubatch,
}
out = root / "results" / "batch-sizing-policy-audit-20260807.json"
out.write_text(json.dumps(payload, indent=2) + "\n")
print(out)
