#!/usr/bin/env python3
"""Audit kernel printk/console policy and Laguna provenance."""
import json
import re
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ENV = (ROOT / "env.sh").read_text(errors="replace")
BENCH = (ROOT / "scripts/bench-serial.sh").read_text(errors="replace")


def read(path):
    try:
        return Path(path).read_text(errors="replace").strip()
    except OSError:
        return None


printk = read("/proc/sys/kernel/printk")
levels = [int(value) for value in printk.split()] if printk else []
cmdline = read("/proc/cmdline") or ""
source = ENV + "\n" + BENCH
control_terms = ("kernel.printk", "console_loglevel", "ignore_loglevel", "loglevel=", "quiet")
metric_terms = ("kernel_printk", "console_loglevel", "kernel_loglevel", "printk_policy")
report = {
    "generated_utc": datetime.now(timezone.utc).isoformat(),
    "angle": "kernel-printk-console-policy",
    "live_policy": {
        "printk_raw": printk,
        "console_loglevel": levels[0] if len(levels) == 4 else None,
        "default_message_loglevel": levels[1] if len(levels) == 4 else None,
        "minimum_console_loglevel": levels[2] if len(levels) == 4 else None,
        "default_console_loglevel": levels[3] if len(levels) == 4 else None,
        "kernel_cmdline": cmdline,
        "cmdline_quiet": bool(re.search(r"(?:^|\s)quiet(?:\s|$)", cmdline)),
        "cmdline_loglevel": bool(re.search(r"(?:^|\s)(?:loglevel=\d+|ignore_loglevel)(?:\s|$)", cmdline)),
    },
    "canonical_policy": {
        "env_or_harness_controls_printk": [term for term in control_terms if term in source],
        "metrics_record_printk_policy": [term for term in metric_terms if f'"{term}"' in BENCH],
    },
}
assert len(levels) == 4
assert all(0 <= value <= 15 for value in levels)
assert not report["canonical_policy"]["env_or_harness_controls_printk"]
assert not report["canonical_policy"]["metrics_record_printk_policy"]
out = ROOT / "results/kernel-printk-console-policy-audit-20260807.json"
out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
print(out)
