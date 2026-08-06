#!/usr/bin/env python3
"""Audit whether Laguna preserves repetition-level benchmark dispersion."""
import json
import os
import re
import subprocess
from pathlib import Path

root = Path(__file__).resolve().parents[1]
env_cmd = f"source {root / 'env.sh'} >/dev/null 2>&1; printf '%s' \"$LX_LLAMA_BENCH\""
bench = subprocess.check_output(["bash", "-lc", env_cmd], text=True)
help_text = subprocess.run(
    ["bash", "-lc", f"source {root / 'env.sh'} >/dev/null 2>&1; exec \"$LX_LLAMA_BENCH\" --help"],
    text=True,
    stdout=subprocess.PIPE,
    stderr=subprocess.STDOUT,
    check=False,
).stdout
harness = (root / "scripts/bench-serial.sh").read_text()

logs = sorted((root / "results").glob("**/llama-bench.log"))
parsed_rows = 0
rows_with_samples = 0
rows_with_stddev = 0
sample_fields = {"samples_ts", "samples_ns", "samples"}
stddev_fields = {"stddev_ts", "stddev", "stddev_tokens_per_second"}
for log in logs:
    text = log.read_text(errors="replace")
    for match in re.finditer(r"\[\s*\{.*?\}\s*\]", text, re.DOTALL):
        try:
            rows = json.loads(match.group(0))
        except json.JSONDecodeError:
            continue
        for row in rows:
            if not isinstance(row, dict) or "avg_ts" not in row:
                continue
            parsed_rows += 1
            rows_with_samples += bool(sample_fields & row.keys())
            rows_with_stddev += bool(stddev_fields & row.keys())

artifact = {
    "executable": bench,
    "repetitions_supported": "--repetitions" in help_text,
    "configured_repetitions": int(os.environ.get("LX_REPS", "5")),
    "harness_requests_repetitions": '"$LX_REPS"' in harness and "-r" in harness,
    "harness_parser_uses_mean_only": 'r.get("avg_ts")' in harness,
    "metrics_records_repetition_count": '"reps": int("$LX_REPS")' in harness,
    "metrics_records_dispersion": any(field in harness for field in stddev_fields | sample_fields),
    "historical_logs_scanned": len(logs),
    "parsed_benchmark_rows": parsed_rows,
    "rows_with_samples": rows_with_samples,
    "rows_with_stddev": rows_with_stddev,
}
out = root / "results/benchmark-dispersion-provenance-audit-20260807.json"
out.write_text(json.dumps(artifact, indent=2) + "\n")
print(out)
print(json.dumps(artifact, indent=2))
