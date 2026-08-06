#!/usr/bin/env python3
"""Reconcile the active champion claim against primary artifacts only."""
import json
import re
from pathlib import Path

root = Path(__file__).resolve().parents[1]
status_path = root / "results/MOUNT_DOOM_STATUS.md"
latest_path = root / "results/LATEST_SCORE.json"
champion_dir = root / "results/20260731T141436Z"
champion_path = champion_dir / "score.json"
status = status_path.read_text()
latest = json.loads(latest_path.read_text())
champion = json.loads(champion_path.read_text())

literal_target = 2.0
status_target_match = re.search(r"Goal \| score \*\*≥ ([0-9.]+)", status)
assert status_target_match, "status-board target not found"
status_target = float(status_target_match.group(1))
quality_artifacts = sorted(
    p.name for p in champion_dir.iterdir()
    if any(term in p.name.lower() for term in ("golden", "quality", "ppl", "perplex", "logit"))
)
score = champion["score"]
report = {
    "audit": "mandatory_primary_evidence_checkpoint",
    "costly_action_launched": False,
    "champion": {
        "stamp": champion["candidate_meta"]["stamp"],
        "score": score,
        "decode_speedup": champion["decode_speedup"],
        "prefill_speedup": champion["prefill_speedup"],
        "floors_ok": champion["floors_ok"],
        "primary_directory_files": sorted(p.name for p in champion_dir.iterdir()),
        "quality_artifacts": quality_artifacts,
        "complete_quality_parity_supported_by_primary_directory": bool(quality_artifacts),
    },
    "latest": {
        "stamp": latest["candidate_meta"]["stamp"],
        "score": latest["score"],
        "below_champion_percent": (1.0 - latest["score"] / score) * 100.0,
    },
    "target_reconciliation": {
        "status_board_target": status_target,
        "literal_user_target": literal_target,
        "targets_contradict": status_target != literal_target,
        "champion_absolute_gap_to_literal_target": literal_target - score,
        "champion_multiplicative_improvement_required_percent": (literal_target / score - 1.0) * 100.0,
    },
    "measured_negative_results": {
        "latest_does_not_reproduce_champion": latest["score"] < score,
        "champion_does_not_meet_literal_target": score < literal_target,
        "champion_primary_directory_has_no_quality_artifact": not quality_artifacts,
    },
}
assert report["target_reconciliation"]["targets_contradict"]
assert report["measured_negative_results"]["latest_does_not_reproduce_champion"]
assert report["measured_negative_results"]["champion_does_not_meet_literal_target"]
assert report["measured_negative_results"]["champion_primary_directory_has_no_quality_artifact"]
out = root / "results/mandatory-primary-evidence-checkpoint-20260807.json"
out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
print(out)
