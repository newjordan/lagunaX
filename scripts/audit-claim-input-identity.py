#!/usr/bin/env python3
"""Audit whether active Laguna speed claims cryptographically identify their inputs."""

import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BASELINE_PATH = ROOT / "baseline/baseline.json"
LATEST_PATH = ROOT / "results/LATEST_SCORE.json"
CHAMPION_PATH = ROOT / "results/20260731T141436Z/metrics.json"
OUT_PATH = ROOT / "results/claim-input-identity-audit-20260807.json"
LITERAL_TARGET = 2.0

baseline = json.loads(BASELINE_PATH.read_text())
latest = json.loads(LATEST_PATH.read_text())
champion = json.loads(CHAMPION_PATH.read_text())
records = {"baseline": baseline, "latest": latest["candidate_meta"], "champion": champion}

required_hash_fields = ("binary_sha256", "model_sha256")
missing = {
    name: [field for field in required_hash_fields if not record.get(field)]
    for name, record in records.items()
}
assert all(fields == list(required_hash_fields) for fields in missing.values())
assert baseline["model"] == latest["candidate_meta"]["model"] == champion["model"]
assert latest["candidate_meta"]["binary"] == champion["binary"]

# Hash the small primary JSON records so this reconciliation itself is immutable;
# do not hash multi-GB model/binary inputs during this read-only checkpoint.
def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()

champion_score = (champion["tg128"] / baseline["tg128"]) ** 0.75 * (
    champion["pp512"] / baseline["pp512"]
) ** 0.25
latest_score = float(latest["score"])
assert abs(champion_score - 1.227) < 0.001
assert latest_score < champion_score < LITERAL_TARGET

artifact = {
    "audit": "claim_input_cryptographic_identity",
    "primary_record_sha256": {
        str(BASELINE_PATH): sha256(BASELINE_PATH),
        str(LATEST_PATH): sha256(LATEST_PATH),
        str(CHAMPION_PATH): sha256(CHAMPION_PATH),
    },
    "required_input_hash_fields": list(required_hash_fields),
    "missing_input_hash_fields": missing,
    "all_claim_inputs_cryptographically_identified": False,
    "path_strings_agree": {
        "model_across_all_records": True,
        "candidate_binary_latest_vs_champion": True,
    },
    "latest_score": latest_score,
    "recomputed_champion_score": champion_score,
    "records_agree": False,
    "literal_target": LITERAL_TARGET,
    "best_active_score": champion_score,
    "absolute_delta_to_literal_target": LITERAL_TARGET - champion_score,
    "relative_improvement_required_pct": (LITERAL_TARGET / champion_score - 1.0) * 100.0,
    "measured_negative_results": [
        "baseline, latest candidate, and champion records omit binary_sha256 and model_sha256",
        "LATEST_SCORE.json does not identify the best active score",
        "the recomputed champion remains below the literal 2.0 objective",
    ],
    "completion_supported": False,
}
OUT_PATH.write_text(json.dumps(artifact, indent=2) + "\n")
print(OUT_PATH)
