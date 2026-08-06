#!/usr/bin/env python3
"""Audit benchmark stderr persistence and output serialization policy."""
import json
import pathlib
import re

ROOT = pathlib.Path(__file__).resolve().parents[1]
bench = (ROOT / "scripts/bench-serial.sh").read_text()
invocations = [line for line in bench.splitlines() if re.match(r'^(?:PP|TG)_JSON=', line) and '"$LX_LLAMA_BENCH"' in line]
result = {
    "benchmark_invocation_count": len(invocations),
    "stderr_appended_to_raw_log_count": sum('2>>"$RAW_LOG"' in x for x in invocations),
    "stdout_output_format": "json" if re.search(r'^\s*-o json\s*$', bench, re.M) else None,
    "raw_json_appended_after_each_run_count": len(re.findall(r'echo "\$(?:PP|TG)_JSON" >>"\$RAW_LOG"', bench)),
    "raw_log_candidate_location": "$LX_RESULTS/<UTC stamp>/llama-bench.log",
    "timing_contamination_assessment": "stderr writes occur during benchmark processes; stdout JSON is appended only after each process exits",
}
print(json.dumps(result, indent=2, sort_keys=True))
