#!/usr/bin/env python3
"""Audit whether benchmark uncertainty survives into scored metric artifacts."""
import json
from pathlib import Path

root = Path(__file__).resolve().parents[1]
logs = sorted((root / "results").rglob("llama-bench.log"))
rows = []
for path in logs:
    text = path.read_text(errors="replace")
    decoder = json.JSONDecoder()
    for pos, char in enumerate(text):
        if char != "[":
            continue
        try:
            value, _ = decoder.raw_decode(text[pos:])
        except json.JSONDecodeError:
            continue
        if isinstance(value, list) and value and all(isinstance(row, dict) for row in value):
            rows.extend(value)

metric_paths = sorted((root / "results").rglob("metrics.json"))
metrics = []
for path in metric_paths:
    try:
        value = json.loads(path.read_text())
    except (json.JSONDecodeError, OSError):
        continue
    if isinstance(value, dict):
        metrics.append(value)

uncertainty_keys = {"stddev_ns", "stddev_ts", "samples_ns", "samples_ts"}
report = {
    "llama_bench_logs": len(logs),
    "benchmark_rows": len(rows),
    "rows_with_stddev_ts": sum("stddev_ts" in row for row in rows),
    "rows_with_samples_ts": sum("samples_ts" in row for row in rows),
    "metrics_artifacts": len(metrics),
    "metrics_with_any_uncertainty": sum(bool(uncertainty_keys & value.keys()) for value in metrics),
    "scored_metrics_retain_only_means": all(
        not (uncertainty_keys & value.keys()) for value in metrics
    ),
}
assert report["benchmark_rows"] > 0
assert report["rows_with_stddev_ts"] > 0
assert report["rows_with_samples_ts"] > 0
assert report["metrics_artifacts"] > 0
assert report["metrics_with_any_uncertainty"] == 0
out = root / "results" / "benchmark-uncertainty-audit-20260807.json"
out.write_text(json.dumps(report, indent=2) + "\n")
print(json.dumps(report, indent=2))
