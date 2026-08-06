#!/usr/bin/env python3
import json
from pathlib import Path

root = Path(__file__).resolve().parents[1]
latest_pointer = (root / "results/LATEST_DIR.txt").read_text().strip().rstrip("/")
latest = json.loads((root / "results/LATEST_SCORE.json").read_text())
champion_path = root / "results/20260731T141436Z/score.json"
champion = json.loads(champion_path.read_text())
status = (root / "results/MOUNT_DOOM_STATUS.md").read_text()

target = 2.0
report = {
    "audit": "active_evidence_reconciliation",
    "latest_pointer": latest_pointer,
    "latest_stamp": latest["candidate_meta"]["stamp"],
    "latest_score": latest["score"],
    "champion_stamp": champion["candidate_meta"]["stamp"],
    "champion_score": champion["score"],
    "champion_quality_claim": "golden OK" if "golden | — | OK | **OK**" in status else "not found",
    "latest_is_champion": latest["candidate_meta"]["stamp"] == champion["candidate_meta"]["stamp"],
    "latest_vs_champion_absolute": latest["score"] - champion["score"],
    "latest_vs_champion_percent": (latest["score"] / champion["score"] - 1.0) * 100.0,
    "literal_target_score": target,
    "champion_absolute_gap_to_target": target - champion["score"],
    "champion_additional_multiplicative_improvement_percent": (target / champion["score"] - 1.0) * 100.0,
    "costly_action_launched": False,
}
assert latest_pointer.endswith(latest["candidate_meta"]["stamp"])
assert report["champion_quality_claim"] == "golden OK"
assert not report["latest_is_champion"]
assert report["latest_vs_champion_absolute"] < 0
assert report["champion_absolute_gap_to_target"] > 0
out = root / "results/active-evidence-reconciliation-audit-20260807.json"
out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
print(out)
