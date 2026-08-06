#!/usr/bin/env python3
"""Audit whether the recorded context size actually reaches llama-bench."""
import json
import os
import re
import subprocess
from pathlib import Path

root = Path(__file__).resolve().parents[1]
env_text = (root / "env.sh").read_text()
bench_text = (root / "scripts" / "bench-serial.sh").read_text()

m = re.search(r'export CTX="\$\{CTX:-([0-9]+)\}"', env_text)
assert m, "CTX default not found"
recorded_default = int(m.group(1))
common = bench_text.split("COMMON=(", 1)[1].split("\n  )", 1)[0]
invocations = [line.strip() for line in bench_text.splitlines() if '"$LX_LLAMA_BENCH" "${COMMON[@]}"' in line]

probe = subprocess.run(
    ["bash", "-lc", 'source ./env.sh >/dev/null 2>&1; "$LX_LLAMA_BENCH" --help'],
    cwd=root, text=True, capture_output=True, check=True,
)
help_text = probe.stdout + probe.stderr
ctx_option = bool(re.search(r'(^|\s)(-c|--ctx-size|--ctx)(\s|[ ,<])', help_text))

report = {
    "direction": "effective context-size provenance",
    "recorded_ctx_default": recorded_default,
    "ctx_recorded_in_metrics": '"ctx": int("$CTX")' in bench_text,
    "ctx_passed_in_common_args": bool(re.search(r'(^|\s)(-c|--ctx-size|--ctx)(\s|$)', common)),
    "ctx_passed_in_benchmark_invocations": any(re.search(r'(^|\s)(-c|--ctx-size|--ctx)(\s|$)', line) for line in invocations),
    "active_executable_exposes_ctx_option": ctx_option,
    "benchmark_invocation_count": len(invocations),
    "effective_window_controls": {"prefill": "-p $LX_PP -n 0", "decode": "-p 0 -n $LX_TG"},
}
out = root / "results" / "effective-context-provenance-audit-20260807.json"
out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
print(out)
