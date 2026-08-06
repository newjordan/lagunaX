#!/usr/bin/env python3
"""Audit whether serial benchmark artifacts preserve the exact executed argv."""
import json
import os
import re
import subprocess
from pathlib import Path

root = Path(__file__).resolve().parents[1]
harness_path = root / "scripts" / "bench-serial.sh"
harness = harness_path.read_text()
env = os.environ.copy()
probe = subprocess.run(
    ["bash", "-lc", 'source ./env.sh >/dev/null 2>&1; "$LX_LLAMA_BENCH" --help'],
    cwd=root,
    text=True,
    stdout=subprocess.PIPE,
    stderr=subprocess.STDOUT,
    check=False,
)
if probe.returncode not in (0, 1) or "--repetitions" not in probe.stdout:
    raise SystemExit("unable to probe active llama-bench command contract")

payload = {
    "audit": "exact-benchmark-command-provenance",
    "executable_help_probed": True,
    "harness_constructs_common_argv_array": "COMMON=(" in harness,
    "separate_prefill_and_decode_invocations": harness.count('"$LX_LLAMA_BENCH" "${COMMON[@]}"') == 2,
    "metrics_records_exact_argv": bool(re.search(r'"(?:argv|command|command_line)"\s*:', harness)),
    "metrics_records_binary_hash": '"binary_sha256"' in harness,
    "risk": "saved metrics cannot reconstruct or mechanically compare the exact effective benchmark command lines",
}
assert payload["harness_constructs_common_argv_array"]
assert payload["separate_prefill_and_decode_invocations"]
assert not payload["metrics_records_exact_argv"]
assert payload["metrics_records_binary_hash"]

out = root / "results" / "exact-benchmark-command-provenance-audit-20260807.json"
out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
print(out)
