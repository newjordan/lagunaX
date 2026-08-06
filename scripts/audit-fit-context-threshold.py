#!/usr/bin/env python3
"""Audit llama-bench's minimum-context threshold used by device-memory fitting."""
import json
import os
import pathlib
import re
import subprocess

ROOT = pathlib.Path(__file__).resolve().parents[1]
env_text = (ROOT / "env.sh").read_text()
harness_text = (ROOT / "scripts/bench-serial.sh").read_text()
env = os.environ.copy()
subprocess.run(["bash", "-lc", f"source {ROOT/'env.sh'}; \"$LX_LLAMA_BENCH\" --help"], cwd=ROOT, env=env,
               text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, check=True)
help_text = subprocess.run(["bash", "-lc", f"source {ROOT/'env.sh'}; \"$LX_LLAMA_BENCH\" --help"], cwd=ROOT, env=env,
                           text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, check=True).stdout
match = re.search(r"-fitc, --fit-ctx <n>\s+minimum ctx size for --fit-target \(default: (\d+)\)", help_text)
if not match:
    raise SystemExit("llama-bench help did not expose the expected --fit-ctx contract")
source_explicit = bool(re.search(r"(?:^|\s)(?:-fitc|--fit-ctx)(?:\s|=)", env_text + "\n" + harness_text))
environment_explicit = any(k in os.environ for k in ("LX_FIT_CTX", "FIT_CTX"))
report = {
    "control": "--fit-ctx",
    "help_default": int(match.group(1)),
    "semantics": "minimum ctx size for --fit-target",
    "active_source_explicit": source_explicit,
    "process_environment_explicit": environment_explicit,
    "effective_active_policy": int(match.group(1)) if not (source_explicit or environment_explicit) else None,
}
out = ROOT / "results/fit-context-threshold-audit-20260807.json"
out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
print(out)
