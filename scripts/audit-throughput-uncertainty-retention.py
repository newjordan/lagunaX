#!/usr/bin/env python3
"""Audit whether the serial score retains benchmark uncertainty evidence."""
import json
import re
from pathlib import Path

root = Path(__file__).resolve().parents[1]
harness = (root / "scripts/bench-serial.sh").read_text()

report = {
    "control": "throughput uncertainty retention",
    "parser_selects_avg_only": bool(re.search(r'avg = r\.get\("avg_ts"\) or r\.get\("avg_tokens_per_second"\)', harness)),
    "parser_prints_single_float": bool(re.search(r'print\(float\(avg\)\)', harness)),
    "metrics_store_point_estimates": all(x in harness for x in ['"pp512": float("$PP_TS")', '"tg128": float("$TG_TS")']),
    "metrics_store_uncertainty": bool(re.search(r'(?i)stddev|confidence[_ -]?interval|ci95|standard[_ -]?error', harness)),
    "raw_benchmark_json_retained_in_log": all(x in harness for x in ['echo "$PP_JSON" >>"$RAW_LOG"', 'echo "$TG_JSON" >>"$RAW_LOG"']),
    "interpretation": "The canonical metrics and score path reduce each workload to avg_ts; uncertainty is not retained as structured score evidence.",
}
if not (report["parser_selects_avg_only"] and report["parser_prints_single_float"] and report["metrics_store_point_estimates"]):
    raise SystemExit("serial metrics contract changed")
out = root / "results/throughput-uncertainty-retention-audit-20260807.json"
out.write_text(json.dumps(report, indent=2) + "\n")
print(out)
