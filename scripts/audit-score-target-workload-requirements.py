#!/usr/bin/env python3
"""Audit workload-specific requirements and sensitivity of the Laguna 2x score."""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BASE = json.loads((ROOT / "baseline/baseline.json").read_text())
SCORE_SOURCE = (ROOT / "scripts/score.py").read_text()

D_EXP = 0.75
P_EXP = 0.25
TARGET = 2.0
base_d = float(BASE["tg128"])
base_p = float(BASE["pp512"])

# Solve d^0.75 * p^0.25 = 2 for representative fixed-workload cases.
def decode_needed(prefill_speedup: float) -> float:
    return (TARGET / (prefill_speedup ** P_EXP)) ** (1.0 / D_EXP)


def prefill_needed(decode_speedup: float) -> float:
    return (TARGET / (decode_speedup ** D_EXP)) ** (1.0 / P_EXP)

report = {
    "audit": "score-target-workload-requirements",
    "target_score": TARGET,
    "formula": "decode_speedup^0.75 * prefill_speedup^0.25",
    "baseline": {"decode_tok_s": base_d, "prefill_tok_s": base_p},
    "requirements": {
        "balanced_speedup_each": TARGET,
        "balanced_decode_tok_s": base_d * TARGET,
        "balanced_prefill_tok_s": base_p * TARGET,
        "decode_speedup_if_prefill_unchanged": decode_needed(1.0),
        "decode_tok_s_if_prefill_unchanged": base_d * decode_needed(1.0),
        "prefill_speedup_if_decode_unchanged": prefill_needed(1.0),
        "prefill_tok_s_if_decode_unchanged": base_p * prefill_needed(1.0),
        "decode_speedup_if_prefill_at_floor": decode_needed(0.95),
        "decode_tok_s_if_prefill_at_floor": base_d * decode_needed(0.95),
    },
    "sensitivity": {
        "score_elasticity_to_decode": D_EXP,
        "score_elasticity_to_prefill": P_EXP,
        "decode_to_prefill_elasticity_ratio": D_EXP / P_EXP,
    },
    "source_checks": {
        "formula_present": '"decode_speedup^0.75 * prefill_speedup^0.25"' in SCORE_SOURCE,
        "decode_floor_present": "DECODE_FLOOR = 0.95" in SCORE_SOURCE,
        "prefill_floor_present": "PREFILL_FLOOR = 0.95" in SCORE_SOURCE,
    },
}
assert all(report["source_checks"].values())
assert abs(report["requirements"]["decode_speedup_if_prefill_unchanged"] - TARGET ** (4 / 3)) < 1e-12
assert report["sensitivity"]["decode_to_prefill_elasticity_ratio"] == 3.0

out = ROOT / "benchmark/results/score-target-workload-requirements-audit-20260807.json"
out.parent.mkdir(parents=True, exist_ok=True)
out.write_text(json.dumps(report, indent=2) + "\n")
print(out)
