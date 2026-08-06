#!/usr/bin/env python3
"""Audit live RCU callback/runtime policy and Laguna provenance."""
import json
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ENV = (ROOT / "env.sh").read_text(errors="replace")
BENCH = (ROOT / "scripts/bench-serial.sh").read_text(errors="replace")
CMDLINE = Path("/proc/cmdline").read_text(errors="replace").strip()
PARAMETERS = {
    "use_softirq": "/sys/module/rcutree/parameters/use_softirq",
    "kthread_prio": "/sys/module/rcutree/parameters/kthread_prio",
    "rcu_resched_ns": "/sys/module/rcutree/parameters/rcu_resched_ns",
    "rcu_nocb_gp_stride": "/sys/module/rcutree/parameters/rcu_nocb_gp_stride",
    "rcu_expedited": "/sys/module/rcupdate/parameters/rcu_expedited",
    "rcu_normal": "/sys/module/rcupdate/parameters/rcu_normal",
    "rcu_normal_after_boot": "/sys/module/rcupdate/parameters/rcu_normal_after_boot",
    "rcu_cpu_stall_timeout": "/sys/module/rcupdate/parameters/rcu_cpu_stall_timeout",
}

def read(path):
    try:
        return Path(path).read_text(errors="replace").strip()
    except OSError:
        return None

live = {name: read(path) for name, path in PARAMETERS.items()}
terms = tuple(PARAMETERS) + ("rcu_nocbs", "rcu_nocb_poll", "rcu_nocb_gp_stride")
source = ENV + "\n" + BENCH
report = {
    "generated_utc": datetime.now(timezone.utc).isoformat(),
    "angle": "kernel-rcu-runtime-policy",
    "live_policy": live,
    "kernel_cmdline": CMDLINE,
    "cmdline_rcu_controls": [term for term in terms if term in CMDLINE],
    "canonical_policy": {
        "env_or_harness_controls_rcu": [term for term in terms if term in source],
        "metrics_record_rcu": [term for term in terms if f'\"{term}\"' in BENCH],
    },
}
assert all(value is not None for value in live.values())
assert not report["cmdline_rcu_controls"]
assert not report["canonical_policy"]["env_or_harness_controls_rcu"]
assert not report["canonical_policy"]["metrics_record_rcu"]
out = ROOT / "results/kernel-rcu-runtime-policy-audit-20260807.json"
out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
print(out)
