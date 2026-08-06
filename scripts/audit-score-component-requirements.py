#!/usr/bin/env python3
"""Quantify component throughput required to reach the serial 2x score."""
import json
from pathlib import Path

root = Path(__file__).resolve().parents[1]
score = json.loads((root / "results/LATEST_SCORE.json").read_text())
target = 2.0
dexp, pexp = 0.75, 0.25
d, p = score["decode_speedup"], score["prefill_speedup"]
required_decode_if_prefill_fixed = (target / (p ** pexp)) ** (1 / dexp)
required_prefill_if_decode_fixed = (target / (d ** dexp)) ** (1 / pexp)
required_balanced_multiplier = target / score["score"]
out = {
    "audit": "score-component-requirements",
    "target_score": target,
    "current_score": score["score"],
    "current_decode_speedup": d,
    "current_prefill_speedup": p,
    "required_decode_speedup_if_prefill_fixed": required_decode_if_prefill_fixed,
    "required_decode_tok_s_if_prefill_fixed": required_decode_if_prefill_fixed * score["baseline_decode_tok_s"],
    "required_prefill_speedup_if_decode_fixed": required_prefill_if_decode_fixed,
    "required_prefill_tok_s_if_decode_fixed": required_prefill_if_decode_fixed * score["baseline_prefill_tok_s"],
    "required_equal_multiplier_on_both_current_components": required_balanced_multiplier,
    "formula": score["formula"],
}
path = root / "benchmark/results/score-component-requirements-audit-20260807.json"
path.parent.mkdir(parents=True, exist_ok=True)
path.write_text(json.dumps(out, indent=2) + "\n")
print(path)
