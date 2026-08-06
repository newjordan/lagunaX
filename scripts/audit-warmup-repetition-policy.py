#!/usr/bin/env python3
"""Audit warmup and repetition policy in the active serial Laguna benchmark."""
import json
import re
import subprocess
from pathlib import Path

root = Path(__file__).resolve().parents[1]
env_text = (root / "env.sh").read_text()
harness_text = (root / "scripts/bench-serial.sh").read_text()
run = subprocess.run(
    ["bash", "-c", 'source "$1" && "$LX_LLAMA_BENCH" --help', "audit", str(root / "env.sh")],
    text=True, capture_output=True,
)
help_text = run.stdout + run.stderr
if run.returncode not in (0, 1):
    raise SystemExit(f"llama-bench --help failed ({run.returncode})")
rep_default = re.search(r"--repetitions <n>\s+number of times to repeat each test \(default: (\d+)\)", help_text)
if not rep_default or "--no-warmup" not in help_text:
    raise SystemExit("warmup/repetition help contract not found")
env_rep = re.search(r'export LX_REPS="\$\{LX_REPS:-(\d+)\}"', env_text)
if not env_rep:
    raise SystemExit("LX_REPS default not found")
report = {
    "binary_default_repetitions": int(rep_default.group(1)),
    "active_environment_repetitions": int(env_rep.group(1)),
    "serial_harness_passes_repetitions": bool(re.search(r"(?:^|\s)-r\s+\"?\$LX_REPS", harness_text, re.M)),
    "binary_exposes_no_warmup": True,
    "serial_harness_disables_warmup": "--no-warmup" in harness_text,
    "effective_warmup": "enabled" if "--no-warmup" not in harness_text else "disabled",
    "interpretation": "Each separately launched prefill and decode process performs an unrecorded warmup before its measured repetitions.",
}
out = root / "results/warmup-repetition-policy-audit-20260807.json"
out.write_text(json.dumps(report, indent=2) + "\n")
print(out)
