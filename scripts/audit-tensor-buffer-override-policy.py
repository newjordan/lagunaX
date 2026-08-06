#!/usr/bin/env python3
"""Audit llama-bench per-tensor buffer override support and Laguna provenance."""
import json
import os
import pathlib
import subprocess

root = pathlib.Path(__file__).resolve().parents[1]
env_text = (root / "env.sh").read_text()
harness_text = (root / "scripts/bench-serial.sh").read_text()
bench = os.environ.get("LX_LLAMA_BENCH")
if not bench:
    cmd = f"source {root / 'env.sh'} >/dev/null 2>&1; printf %s \"$LX_LLAMA_BENCH\""
    bench = subprocess.check_output(["bash", "-lc", cmd], text=True)
cmd = f"source {root / 'env.sh'} >/dev/null 2>&1; \"$LX_LLAMA_BENCH\" --help 2>&1"
help_run = subprocess.run(["bash", "-lc", cmd], text=True, stdout=subprocess.PIPE)
help_text = help_run.stdout
support_line = next((line.strip() for line in help_text.splitlines()
                     if "--override-tensor" in line), None)
result = {
    "audit": "tensor-buffer-override-policy",
    "llama_bench": bench,
    "cli_support": support_line is not None,
    "cli_evidence": support_line,
    "env_controls_override_tensor": "LX_OVERRIDE_TENSOR" in env_text,
    "harness_forwards_override_tensor": "--override-tensor" in harness_text,
    "harness_records_override_tensor": "override_tensor" in harness_text,
    "conclusion": "supported-but-uncontrolled" if support_line and
        "LX_OVERRIDE_TENSOR" not in env_text and "--override-tensor" not in harness_text
        else "not-supported-or-already-controlled",
}
out = root / "benchmark/results/tensor-buffer-override-policy-audit-20260807.json"
out.parent.mkdir(parents=True, exist_ok=True)
out.write_text(json.dumps(result, indent=2) + "\n")
print(out)
print(json.dumps(result, indent=2))
if result["conclusion"] != "supported-but-uncontrolled":
    raise SystemExit(1)
