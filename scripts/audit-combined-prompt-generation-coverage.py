#!/usr/bin/env python3
"""Audit whether the active serial score covers combined prompt+generation execution."""
import json
import os
import pathlib
import re
import subprocess

ROOT = pathlib.Path(__file__).resolve().parents[1]
ENV = ROOT / "env.sh"
HARNESS = ROOT / "scripts" / "bench-serial.sh"
OUT = ROOT / "results" / "combined-prompt-generation-coverage-audit-20260807.json"
text = HARNESS.read_text()

# Resolve the executable through the same environment used by the harness.
cmd = f'source {ENV!s} >/dev/null 2>&1; printf "%s" "$LX_LLAMA_BENCH"'
bench = subprocess.check_output(["bash", "-lc", cmd], text=True)
help_run = subprocess.run(
    ["bash", "-lc", f'source {ENV!s} >/dev/null 2>&1; "$LX_LLAMA_BENCH" --help'],
    text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, check=False
)
help_text = help_run.stdout

invocations = []
for line_no, line in enumerate(text.splitlines(), 1):
    if '"$LX_LLAMA_BENCH"' not in line or not re.search(r'(^|\s)-(?:p|n)\s', line):
        continue
    p = re.search(r'-p\s+("?[^ ]+"?)\s+-n\s+("?[^ )]+"?)', line)
    invocations.append({"line": line_no, "prompt_argument": p.group(1) if p else None,
                        "generation_argument": p.group(2) if p else None,
                        "uses_combined_pg_flag": "-pg" in line})

report = {
    "audit": "combined-prompt-generation-workload-coverage",
    "executable": bench,
    "executable_supports_combined_pg": any(line.strip().startswith("-pg ") for line in help_text.splitlines()),
    "combined_pg_help": next((line.strip() for line in help_text.splitlines() if line.strip().startswith("-pg ")), None),
    "benchmark_invocations": invocations,
    "invocation_count": len(invocations),
    "combined_invocation_count": sum(i["uses_combined_pg_flag"] for i in invocations),
    "separate_only_policy": len(invocations) > 0 and all(not i["uses_combined_pg_flag"] for i in invocations),
}
OUT.write_text(json.dumps(report, indent=2) + "\n")
print(OUT)
