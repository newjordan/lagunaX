#!/usr/bin/env python3
"""Audit kernel taint and CPU microcode provenance without running Laguna."""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

def read(path: str):
    try:
        return Path(path).read_text().strip()
    except (OSError, UnicodeError):
        return None

cpuinfo = read("/proc/cpuinfo") or ""
microcodes = sorted({line.split(":", 1)[1].strip() for line in cpuinfo.splitlines() if line.lower().startswith("microcode") and ":" in line})
taint_raw = read("/proc/sys/kernel/tainted")
taint = int(taint_raw) if taint_raw is not None else None
cmdline = read("/proc/cmdline")
env_text = (ROOT / "env.sh").read_text()
harness_text = (ROOT / "scripts/bench-serial.sh").read_text()
latest = json.loads((ROOT / "results/LATEST_SCORE.json").read_text())
champion = json.loads((ROOT / "results/20260731T141436Z/score.json").read_text())
status = (ROOT / "results/MOUNT_DOOM_STATUS.md").read_text()
needle_text = (env_text + "\n" + harness_text).lower()
report = {
    "audit": "host_kernel_taint_microcode_policy",
    "host": {
        "kernel_taint_mask": taint,
        "kernel_is_tainted": None if taint is None else taint != 0,
        "cpu_microcode_revisions": microcodes,
        "kernel_command_line": cmdline,
    },
    "laguna": {
        "configures_microcode_or_taint_policy": any(x in needle_text for x in ("microcode", "kernel.tainted", "/proc/sys/kernel/tainted")),
        "records_microcode_or_taint_metrics": any(x in harness_text.lower() for x in ("microcode", "kernel_taint", "kernel.tainted")),
        "rejects_tainted_kernel": "kernel_is_tainted" in harness_text.lower() or "taint_mask" in harness_text.lower(),
    },
    "evidence_reconciliation": {
        "champion_score": champion["score"],
        "champion_quality_claim": "golden OK" if "golden | — | OK | **OK**" in status else "not found",
        "latest_score": latest["score"],
        "latest_vs_champion_percent": (latest["score"] / champion["score"] - 1.0) * 100.0,
        "literal_target_score": 2.0,
        "champion_additional_multiplicative_improvement_percent": (2.0 / champion["score"] - 1.0) * 100.0,
        "costly_action_launched": False,
    },
}
assert taint is not None
assert microcodes
assert report["evidence_reconciliation"]["champion_quality_claim"] == "golden OK"
assert report["evidence_reconciliation"]["latest_vs_champion_percent"] < 0
out = ROOT / "benchmark/results/host-kernel-taint-microcode-policy-audit-20260807.json"
out.parent.mkdir(parents=True, exist_ok=True)
out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
print(out)
