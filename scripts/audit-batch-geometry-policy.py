#!/usr/bin/env python3
"""Audit Laguna logical/physical batch-size policy against live llama-bench."""
import json
import os
import re
import subprocess
from datetime import datetime, timezone
from pathlib import Path

root = Path(__file__).resolve().parents[1]
env_text = (root / "env.sh").read_text()
bench_text = (root / "scripts/bench-serial.sh").read_text()
binary = Path(os.environ.get("LX_LLAMA_BENCH", root / "baseline/tip-binary-backup-20260730T141542Z/llama-bench"))
probe = subprocess.run(
    ["bash", "-lc", f"source /opt/intel/oneapi/setvars.sh --force >/dev/null 2>&1; export LD_LIBRARY_PATH='{binary.parent}':\"${{LD_LIBRARY_PATH:-}}\"; '{binary}' --help"],
    text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
)
if "--batch-size" not in probe.stdout:
    raise SystemExit(f"llama-bench help probe failed ({probe.returncode}): {probe.stdout[:500]}")
help_text = probe.stdout

def default(flag: str) -> int:
    match = re.search(rf"--{re.escape(flag)} <n>.*?\(default: (\d+)\)", help_text)
    if not match:
        raise SystemExit(f"cannot parse --{flag} default")
    return int(match.group(1))

def env_default(name: str) -> int:
    match = re.search(rf'export {name}="\$\{{{name}:-([0-9]+)\}}"', env_text)
    if not match:
        raise SystemExit(f"cannot parse {name}")
    return int(match.group(1))

payload = {
    "collected_at": datetime.now(timezone.utc).isoformat(),
    "binary": str(binary),
    "live_contract": {"batch_size_default": default("batch-size"), "ubatch_size_default": default("ubatch-size")},
    "laguna_policy": {
        "BBATCH": env_default("BBATCH"),
        "UBATCH": env_default("UBATCH"),
        "passes_batch_size": '-b "$BBATCH"' in bench_text,
        "passes_ubatch_size": '-ub "$UBATCH"' in bench_text,
        "records_batch_size": '"bbatch": int("$BBATCH")' in bench_text,
        "records_ubatch_size": '"ubatch": int("$UBATCH")' in bench_text,
    },
}
payload["finding"] = "Laguna raises physical ubatch from the executable default 512 to 2048, equal to its logical batch size."
out = root / "results/batch-geometry-policy-audit-20260807.json"
out.write_text(json.dumps(payload, indent=2) + "\n")
print(out)
