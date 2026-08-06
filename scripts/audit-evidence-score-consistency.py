#!/usr/bin/env python3
"""Reconcile active Laguna score records against the literal 2x objective."""

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCORE_PATH = ROOT / "results/LATEST_SCORE.json"
STATUS_PATH = ROOT / "results/MOUNT_DOOM_STATUS.md"
OUT_PATH = ROOT / "results/evidence-score-consistency-audit-20260807.json"
LITERAL_TARGET = 2.0

score = json.loads(SCORE_PATH.read_text())
status = STATUS_PATH.read_text()
latest = float(score["score"])
status_champion = 1.227
status_goal = 1.250

assert 'score | 1.0 | 1.209 | **1.227 (+22.74%)**' in status
assert 'Goal | score **≥ 1.250**' in status
assert score["floors_ok"] is True
assert latest < status_champion < LITERAL_TARGET

artifact = {
    "audit": "active_score_record_consistency",
    "sources": [str(SCORE_PATH), str(STATUS_PATH)],
    "latest_score_json": latest,
    "status_board_champion": status_champion,
    "status_board_interim_goal": status_goal,
    "records_agree": latest == status_champion,
    "quality_floors_ok_for_latest_score": bool(score["floors_ok"]),
    "literal_target": LITERAL_TARGET,
    "latest_absolute_delta_to_literal_target": LITERAL_TARGET - latest,
    "latest_relative_improvement_required_pct": (LITERAL_TARGET / latest - 1.0) * 100.0,
    "champion_absolute_delta_to_literal_target": LITERAL_TARGET - status_champion,
    "champion_relative_improvement_required_pct": (LITERAL_TARGET / status_champion - 1.0) * 100.0,
    "measured_negative_results": [
        "LATEST_SCORE.json is below the status-board champion",
        "both active score records remain below the literal 2.0 objective",
    ],
    "completion_supported": False,
}
OUT_PATH.write_text(json.dumps(artifact, indent=2) + "\n")
print(OUT_PATH)
