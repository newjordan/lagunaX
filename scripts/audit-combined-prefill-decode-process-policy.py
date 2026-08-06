#!/usr/bin/env python3
"""Audit whether Laguna uses llama-bench's combined prompt+generation test mode."""
import json
import os
import re
import subprocess
from datetime import datetime, timezone
from pathlib import Path

root = Path(__file__).resolve().parents[1]
env = (root / "env.sh").read_text()
bench = (root / "scripts/bench-serial.sh").read_text()
match = re.search(r'^export LX_LLAMA_BENCH=', env, re.M)
if not match:
    raise SystemExit("LX_LLAMA_BENCH not found")
probe = subprocess.run(
    ["bash", "-lc", 'source "$1/env.sh" && "$LX_LLAMA_BENCH" --help', "audit", str(root)],
    text=True, capture_output=True, check=True,
)
help_text = probe.stdout
binary_probe = subprocess.run(
    ["bash", "-lc", 'source "$1/env.sh" && printf %s "$LX_LLAMA_BENCH"', "audit", str(root)],
    text=True, capture_output=True, check=True,
)
binary = Path(binary_probe.stdout)
pg = re.search(r"-pg <pp,tg>\s+\(default: ([^)]*)\)", help_text)
if not pg:
    raise SystemExit("active llama-bench does not expose -pg")
invocations = re.findall(r'PP_JSON="\$\("\$LX_LLAMA_BENCH"|TG_JSON="\$\("\$LX_LLAMA_BENCH"', bench)
payload = {
    "generated_utc": datetime.now(timezone.utc).isoformat(),
    "binary": str(binary),
    "live_contract": {"flag": "-pg", "syntax": "<pp,tg>", "default": pg.group(1)},
    "active_harness": {
        "llama_bench_timed_invocations": len(invocations),
        "passes_combined_pg": bool(re.search(r'(^|\s)-pg(\s|$)', bench, re.M)),
        "prefill_and_decode_are_separate_processes": len(invocations) == 2,
    },
    "consequence": "separate processes repeat executable startup and model loading; -pg can run both test shapes in one process",
}
assert payload["active_harness"] == {
    "llama_bench_timed_invocations": 2,
    "passes_combined_pg": False,
    "prefill_and_decode_are_separate_processes": True,
}
out = root / "results" / "combined-prefill-decode-process-policy-audit-20260807.json"
out.write_text(json.dumps(payload, indent=2) + "\n")
print(out)
