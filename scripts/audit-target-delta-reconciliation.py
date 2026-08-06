#!/usr/bin/env python3
"""Reconcile the latest serial score with the literal 100%-better target."""
import json
from pathlib import Path

root = Path(__file__).resolve().parents[1]
score_path = root / "results" / "LATEST_SCORE.json"
out_path = root / "results" / "target-delta-reconciliation-20260807.json"
data = json.loads(score_path.read_text())
target = 2.0
current = float(data["score"])
result = {
    "source": str(score_path),
    "track": data["track"],
    "formula": data["formula"],
    "quality_guard_available": "quality" in data or "quality_parity" in data,
    "target_speedup": target,
    "current_speedup": current,
    "target_increase_pct": 100.0,
    "current_increase_pct": float(data["increase_pct"]),
    "absolute_speedup_gap": target - current,
    "additional_gain_required_over_current_pct": (target / current - 1.0) * 100.0,
    "decode_speedup": float(data["decode_speedup"]),
    "prefill_speedup": float(data["prefill_speedup"]),
    "decode_gap_to_2x": target - float(data["decode_speedup"]),
    "prefill_gap_to_2x": target - float(data["prefill_speedup"]),
    "target_met": current >= target,
    "reconciled_candidate": data["candidate_path"],
    "reconciled_baseline": data["baseline_path"],
}
out_path.write_text(json.dumps(result, indent=2) + "\n")
print(out_path)
