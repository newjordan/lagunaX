#!/usr/bin/env python3
"""Audit process timer-slack policy and Laguna benchmark provenance."""
import json
import re
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ENV = ROOT / "env.sh"
HARNESS = ROOT / "scripts/bench-serial.sh"
OUT = ROOT / "results/process-timer-slack-policy-audit-20260807.json"


def read(path: str) -> str | None:
    try:
        return Path(path).read_text().strip()
    except (FileNotFoundError, PermissionError):
        return None


slack_raw = read("/proc/self/timerslack_ns")
if slack_raw is None:
    raise RuntimeError("kernel does not export /proc/self/timerslack_ns")
slack_ns = int(slack_raw)
env_text = ENV.read_text(errors="replace")
harness_text = HARNESS.read_text(errors="replace")
source = env_text + "\n" + harness_text
policy_pattern = re.compile(r"timerslack|timer_slack|PR_SET_TIMERSLACK", re.I)
metrics_pattern = re.compile(r"/proc/(?:self|\$\$|[0-9]+)/timerslack_ns|timer_slack_ns", re.I)

report = {
    "angle": "process-timer-slack-policy",
    "generated_at": datetime.now(timezone.utc).isoformat(),
    "live": {
        "audit_process_timer_slack_ns": slack_ns,
        "audit_process_timer_slack_us": slack_ns / 1000,
        "proc_interface": "/proc/self/timerslack_ns",
    },
    "laguna": {
        "env_or_harness_controls_timer_slack": bool(policy_pattern.search(source)),
        "serial_harness_records_timer_slack": bool(metrics_pattern.search(harness_text)),
    },
    "interpretation": (
        "Timer slack permits the kernel to coalesce process timer wakeups within the configured "
        "window. The audit records the inherited value and whether Laguna controls or records it."
    ),
}
assert slack_ns >= 0
assert not report["laguna"]["env_or_harness_controls_timer_slack"]
assert not report["laguna"]["serial_harness_records_timer_slack"]
OUT.parent.mkdir(parents=True, exist_ok=True)
OUT.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
print(OUT)
