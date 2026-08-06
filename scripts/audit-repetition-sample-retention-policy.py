#!/usr/bin/env python3
"""Audit repetition policy and whether emitted evidence retains per-repetition samples."""
import json, os, re, subprocess
from pathlib import Path

root = Path(__file__).resolve().parents[1]
bench_script = (root / "scripts/bench-serial.sh").read_text()
env_text = (root / "env.sh").read_text()
bench = Path(os.environ.get("LX_LLAMA_BENCH", root / "baseline/tip-binary-backup-20260730T141542Z/llama-bench"))
help_text = subprocess.run([str(bench), "--help"], text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, check=True).stdout
m = re.search(r"(?:-r,\s*)?--repetitions <n>.*?default:\s*(\d+)", help_text, re.I)
if not m:
    raise SystemExit("could not probe repetitions contract")
env_m = re.search(r'^\s*export\s+LX_REPS="?\$\{LX_REPS:-([0-9]+)\}"?', env_text, re.M)
configured = int(env_m.group(1)) if env_m else None
passed = bool(re.search(r"-r\s+[\"']?\$LX_REPS", bench_script))
records_count = '"reps": int("$LX_REPS")' in bench_script
retains_samples = bool(re.search(r'"(?:samples|raw_samples|repetitions)"\s*:', bench_script))
artifact = {
    "binary": str(bench),
    "executable_default_repetitions": int(m.group(1)),
    "laguna_configured_repetitions": configured,
    "laguna_passes_repetitions": passed,
    "laguna_records_repetition_count": records_count,
    "laguna_retains_per_repetition_samples": retains_samples,
    "laguna_parser_extracts_only_mean_avg_ts": 'PP_TS="$(parse_ts' in bench_script and 'TG_TS="$(parse_ts' in bench_script,
}
out = root / "results/repetition-sample-retention-audit-20260807.json"
out.write_text(json.dumps(artifact, indent=2) + "\n")
print(out)
print(json.dumps(artifact, indent=2))
if configured is None or not passed or not records_count or retains_samples:
    raise SystemExit("unexpected repetition policy")
