#!/usr/bin/env python3
"""Reconcile the active serial speed claim against immutable binary and quality evidence."""
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
score_path = ROOT / "results" / "LATEST_SCORE.json"
score = json.loads(score_path.read_text())
records = {}
for role in ("baseline", "candidate"):
    metrics_path = Path(score[f"{role}_path"])
    metrics = json.loads(metrics_path.read_text())
    binary_path = Path(metrics["binary"])
    records[role] = {
        "metrics_path": str(metrics_path),
        "recorded_binary_path": str(binary_path),
        "binary_exists_now": binary_path.is_file(),
        "current_binary_sha256": hashlib.sha256(binary_path.read_bytes()).hexdigest() if binary_path.is_file() else None,
        "recorded_binary_sha256": metrics.get("binary_sha256"),
        "quality_evidence_keys": sorted(k for k in metrics if any(term in k.lower() for term in ("quality", "perplexity", "ppl", "logit", "kl", "token_match"))),
    }
report = {
    "source": str(score_path),
    "target_speedup": 2.0,
    "current_speedup": score["score"],
    "absolute_speedup_gap": 2.0 - score["score"],
    "additional_gain_required_over_current_pct": (2.0 / score["score"] - 1.0) * 100.0,
    "target_met": score["score"] >= 2.0,
    "records": records,
    "recorded_paths_differ": records["baseline"]["recorded_binary_path"] != records["candidate"]["recorded_binary_path"],
    "current_binary_hashes_equal": records["baseline"]["current_binary_sha256"] == records["candidate"]["current_binary_sha256"],
    "both_metrics_omit_binary_hash": all(records[r]["recorded_binary_sha256"] is None for r in records),
    "both_metrics_omit_quality_evidence": all(not records[r]["quality_evidence_keys"] for r in records),
}
assert report["recorded_paths_differ"]
assert report["current_binary_hashes_equal"]
assert report["both_metrics_omit_binary_hash"]
assert report["both_metrics_omit_quality_evidence"]
assert not report["target_met"]
out = ROOT / "results" / "active-claim-binary-identity-reconciliation-20260807.json"
out.write_text(json.dumps(report, indent=2) + "\n")
print(out)
