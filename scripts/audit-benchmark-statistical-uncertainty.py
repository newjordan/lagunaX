#!/usr/bin/env python3
"""Audit whether canonical scoring preserves llama-bench uncertainty statistics."""
import json
import re
from pathlib import Path

root = Path(__file__).resolve().parents[1]
bench = (root / "scripts/bench-serial.sh").read_text()
env = (root / "env.sh").read_text()

assert re.search(r'export LX_REPS="\$\{LX_REPS:-5\}"', env)
assert 'rows[0].get("avg_ts")' in bench or 'r.get("avg_ts")' in bench
assert 'stddev_ts' not in bench
assert '"pp512": float("$PP_TS")' in bench
assert '"tg128": float("$TG_TS")' in bench
assert 'confidence' not in bench.lower()

artifact = {
    "audit": "benchmark-statistical-uncertainty-retention",
    "canonical_repetitions": 5,
    "throughput_statistic_retained": "avg_ts",
    "stddev_ts_retained": False,
    "per_repetition_samples_retained": False,
    "confidence_interval_computed": False,
    "score_inputs": ["pp512", "tg128"],
    "conclusion": "The canonical claim reduces five repetitions to means and cannot quantify uncertainty or distinguish speedup from run variance."
}
out = root / "results/benchmark-statistical-uncertainty-audit-20260807.json"
out.write_text(json.dumps(artifact, indent=2) + "\n")
print(out)
