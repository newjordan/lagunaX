#!/usr/bin/env python3
"""Audit active Laguna score/goal claims for cross-artifact consistency."""
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
score_path = ROOT / "results" / "LATEST_SCORE.json"
board_path = ROOT / "results" / "MOUNT_DOOM_STATUS.md"
score = json.loads(score_path.read_text())
board = board_path.read_text()

board_goal = float(re.search(r"Goal \| score \*\*≥ ([0-9.]+)", board).group(1))
board_champion = float(re.search(r"\| score \| 1\.0 \| [0-9.]+ \| \*\*([0-9.]+)", board).group(1))
literal_target = 2.0
active_score = float(score["score"])

report = {
    "sources": [str(score_path), str(board_path)],
    "active_score": active_score,
    "board_champion_score": board_champion,
    "board_goal": board_goal,
    "literal_target": literal_target,
    "active_vs_board_champion_delta": active_score - board_champion,
    "board_goal_vs_literal_target_delta": literal_target - board_goal,
    "absolute_gap_to_literal_target": literal_target - active_score,
    "additional_gain_required_over_active_pct": (literal_target / active_score - 1.0) * 100.0,
    "active_target_met": active_score >= literal_target,
    "board_champion_target_met": board_champion >= literal_target,
    "board_goal_matches_literal_target": board_goal == literal_target,
    "contradictions": {
        "active_score_differs_from_board_champion": active_score != board_champion,
        "board_goal_is_not_literal_2x_target": board_goal != literal_target,
    },
    "measured_negative_results": {
        "active_score_below_literal_target": active_score < literal_target,
        "board_champion_below_literal_target": board_champion < literal_target,
    },
}
assert report["contradictions"]["active_score_differs_from_board_champion"]
assert report["contradictions"]["board_goal_is_not_literal_2x_target"]
assert report["measured_negative_results"]["active_score_below_literal_target"]
assert report["measured_negative_results"]["board_champion_below_literal_target"]

out = ROOT / "results" / "active-claim-ledger-consistency-audit-20260807.json"
out.write_text(json.dumps(report, indent=2) + "\n")
print(out)
