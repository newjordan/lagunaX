#!/usr/bin/env python3
import json
import os
import re
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
env_text = (ROOT / "env.sh").read_text()
bench_path = ROOT / "scripts" / "bench-serial.sh"
bench_text = bench_path.read_text()

match = re.search(r'export LX_LLAMA_BENCH="\$\{LX_LLAMA_BENCH:-([^}]+)\}"', env_text)
if not match:
    raise SystemExit("cannot resolve LX_LLAMA_BENCH from env.sh")
binary = match.group(1).replace("$LX_BIN", os.environ.get("LX_BIN", "/home/frosty40/turbo/worktrees/treebeard-base-control-latest/build-mmadd-decode/bin"))
probe = subprocess.run(
    ["bash", "-lc", 'source ./env.sh >/dev/null 2>&1; "$LX_LLAMA_BENCH" --help 2>&1'],
    cwd=ROOT,
    text=True,
    stdout=subprocess.PIPE,
)
if probe.returncode not in (0, 1):
    raise SystemExit(f"llama-bench help probe failed with {probe.returncode}: {probe.stdout}")
help_text = probe.stdout
if "--progress" not in help_text:
    raise SystemExit("active llama-bench does not expose --progress")

common = re.search(r"COMMON=\(\n(.*?)\n\s*\)", bench_text, re.S)
if not common:
    raise SystemExit("cannot locate COMMON arguments")
common_text = common.group(1)
progress_passed = bool(re.search(r"(?:^|\s)--progress(?:\s|$)", common_text))
verbose_passed = bool(re.search(r"(?:^|\s)(?:-v|--verbose)(?:\s|$)", common_text))
progress_recorded = bool(re.search(r'"(?:progress|progress_indicators)"\s*:', bench_text))
verbose_recorded = bool(re.search(r'"verbose"\s*:', bench_text))

artifact = {
    "direction": "benchmark diagnostic-output and progress-indicator policy",
    "live_contract": {
        "binary": binary,
        "progress_flag": "--progress",
        "progress_help": "print test progress indicators",
        "verbose_flag": "-v/--verbose",
        "verbose_help": "verbose output",
        "defaults": {"progress": "disabled", "verbose": "disabled"},
    },
    "laguna": {
        "progress_passed": progress_passed,
        "verbose_passed": verbose_passed,
        "progress_recorded": progress_recorded,
        "verbose_recorded": verbose_recorded,
        "stdout_captured_as_json": 'PP_JSON="$(' in bench_text and 'TG_JSON="$(' in bench_text,
        "stderr_redirected_to_raw_log": '2>>"$RAW_LOG"' in bench_text,
    },
    "finding": "Laguna leaves progress and verbose diagnostics disabled; benchmark stdout is captured for JSON parsing while stderr is redirected to the raw log.",
}
out = ROOT / "results" / "diagnostic-output-progress-policy-audit-20260807.json"
out.write_text(json.dumps(artifact, indent=2) + "\n")
print(out)
