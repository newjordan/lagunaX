#!/usr/bin/env python3
"""Audit physical microbatch sizing versus llama-bench defaults."""
import json, os, re, subprocess
from pathlib import Path

root = Path(__file__).resolve().parents[1]
env_text = (root / "env.sh").read_text()
bench_text = (root / "scripts/bench-serial.sh").read_text()
env = os.environ.copy()
subprocess.run(["bash", "-lc", f"source {root/'env.sh'}; env -0"], check=True, stdout=subprocess.DEVNULL)
probe = subprocess.run(["bash", "-lc", f"source {root/'env.sh'}; \"$LX_LLAMA_BENCH\" --help 2>&1"], text=True, capture_output=True, check=True).stdout

def default(flag):
    m = re.search(rf"{re.escape(flag)}[^\n]*\(default: ([^)]+)\)", probe)
    return int(m.group(1)) if m else None

def export(name):
    m = re.search(rf'^export {name}="\$\{{{name}:-([0-9]+)\}}"', env_text, re.M)
    return int(m.group(1)) if m else None

payload = {
    "angle": "logical batch versus physical microbatch sizing",
    "llama_bench_defaults": {"batch_size": default("--batch-size"), "ubatch_size": default("--ubatch-size")},
    "laguna_policy": {"batch_size": export("BBATCH"), "ubatch_size": export("UBATCH")},
    "serial_harness_passes": {
        "batch_size": '-b "$BBATCH"' in bench_text,
        "ubatch_size": '-ub "$UBATCH"' in bench_text,
    },
}
payload["laguna_ubatch_is_default"] = payload["laguna_policy"]["ubatch_size"] == payload["llama_bench_defaults"]["ubatch_size"]
payload["laguna_ubatch_default_ratio"] = payload["laguna_policy"]["ubatch_size"] / payload["llama_bench_defaults"]["ubatch_size"]
out = root / "results" / "batch-microbatch-sizing-audit-20260807.json"
out.write_text(json.dumps(payload, indent=2) + "\n")
print(out)
print(json.dumps(payload, indent=2))
