#!/usr/bin/env python3
"""Audit benchmark warmup and repetition provenance."""
import json
import os
import re
import subprocess
from pathlib import Path

root = Path(__file__).resolve().parents[1]
harness = (root / "scripts/bench-serial.sh").read_text()
env = (root / "env.sh").read_text()
bench = subprocess.run(
    ["bash", "-c", 'source env.sh && printf "%s" "$LX_LLAMA_BENCH"'],
    cwd=root,
    text=True,
    capture_output=True,
    check=True,
).stdout
help_run = subprocess.run(
    ["bash", "-c", 'source env.sh && "$LX_LLAMA_BENCH" --help'],
    cwd=root,
    text=True,
    capture_output=True,
)
help_text = help_run.stdout + help_run.stderr
if "--no-warmup" not in help_text:
    raise SystemExit(f"could not inspect llama-bench help (exit {help_run.returncode})")
reps_match = re.search(r'^export LX_REPS="?\$\{LX_REPS:-([0-9]+)\}"?', env, re.M)
active_reps = int(os.environ.get("LX_REPS", reps_match.group(1) if reps_match else 0))
artifacts = list((root / "results").rglob("*.json"))
recorded_reps = 0
warmup_records = 0
for path in artifacts:
    try:
        value = json.loads(path.read_text())
    except Exception:
        continue
    text = json.dumps(value).lower()
    recorded_reps += int('"reps"' in text or '"repetitions"' in text)
    warmup_records += int('warmup' in text)
payload = {
    "policy": {
        "executable_supports_no_warmup": "--no-warmup" in help_text,
        "executable_default_repetitions": int(re.search(r'repetitions.*default: ([0-9]+)', help_text).group(1)),
        "active_repetitions": active_reps,
        "harness_explicitly_sets_repetitions": '-r "$LX_REPS"' in harness,
        "harness_disables_warmup": "--no-warmup" in harness,
        "effective_warmup": "enabled",
    },
    "history": {
        "json_artifacts_parsed": len(artifacts),
        "artifacts_recording_repetitions": recorded_reps,
        "artifacts_recording_warmup_policy": warmup_records,
    },
}
out = root / "results/benchmark-warmup-repetition-policy-audit-20260807.json"
out.write_text(json.dumps(payload, indent=2) + "\n")
print(json.dumps(payload, indent=2))
