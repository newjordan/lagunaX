#!/usr/bin/env python3
"""Audit llama-bench device-memory fitting policy and Laguna provenance."""
import json
import os
import re
import subprocess
from datetime import datetime, timezone
from pathlib import Path

root = Path(__file__).resolve().parents[1]
env_text = (root / "env.sh").read_text()
bench_text = (root / "scripts/bench-serial.sh").read_text()
bench = os.environ.get("LX_LLAMA_BENCH")
if not bench:
    m = re.search(r'export LX_LLAMA_BENCH="\$\{LX_LLAMA_BENCH:-([^\"]+)\}"', env_text)
    bin_match = re.search(r'export LX_BIN="\$\{LX_BIN:-([^\"]+)\}"', env_text)
    if not (m and bin_match):
        raise SystemExit("cannot resolve LX_LLAMA_BENCH")
    bench = m.group(1).replace("$LX_BIN", bin_match.group(1))
help_text = subprocess.run([bench, "--help"], text=True, capture_output=True, check=True).stdout
fit_target = re.search(r'--fit-target <MiB>\s+([^\n]+)', help_text)
fit_ctx = re.search(r'--fit-ctx <n>\s+([^\n]+)', help_text)
if not (fit_target and fit_ctx):
    raise SystemExit("active executable does not expose expected fit controls")
configured = bool(re.search(r'(^|\s)(--fit-target|-fitt|--fit-ctx|-fitc)(\s|$)', env_text + "\n" + bench_text, re.M))
recorded = bool(re.search(r'fit[_-]?(target|ctx)', bench_text, re.I))
payload = {
    "generated_at": datetime.now(timezone.utc).isoformat(),
    "binary": bench,
    "contract": {
        "fit_target": fit_target.group(1).strip(),
        "fit_ctx": fit_ctx.group(1).strip(),
    },
    "laguna": {"configures_fit_controls": configured, "records_fit_controls": recorded},
}
out = root / "results" / "device-memory-fit-policy-audit-20260807.json"
out.write_text(json.dumps(payload, indent=2) + "\n")
print(out)
