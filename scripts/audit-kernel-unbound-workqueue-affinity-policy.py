#!/usr/bin/env python3
"""Audit kernel unbound-workqueue affinity and Laguna provenance."""
import json
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
WQ_ROOT = Path("/sys/devices/virtual/workqueue")
ENV = (ROOT / "env.sh").read_text(errors="replace")
BENCH = (ROOT / "scripts/bench-serial.sh").read_text(errors="replace")


def read(path):
    try:
        return path.read_text(errors="replace").strip()
    except OSError:
        return None


queues = {}
for mask_path in sorted(WQ_ROOT.glob("*/cpumask")):
    queue = mask_path.parent
    queues[queue.name] = {
        "cpumask": read(mask_path),
        "per_cpu": read(queue / "per_cpu"),
        "nice": read(queue / "nice"),
        "max_active": read(queue / "max_active"),
        "affinity_scope": read(queue / "affinity_scope"),
        "affinity_strict": read(queue / "affinity_strict"),
    }

source = ENV + "\n" + BENCH
control_terms = ("workqueue/cpumask", "wq_unbound_cpumask", "WORKQUEUE_CPUMASK")
metric_terms = ("workqueue_cpumask", "workqueue_cpumask_isolated", "workqueue_affinity")
report = {
    "generated_utc": datetime.now(timezone.utc).isoformat(),
    "angle": "kernel-unbound-workqueue-affinity-policy",
    "live_policy": {
        "online_cpus": read(Path("/sys/devices/system/cpu/online")),
        "global_cpumask": read(WQ_ROOT / "cpumask"),
        "requested_cpumask": read(WQ_ROOT / "cpumask_requested"),
        "isolated_cpumask": read(WQ_ROOT / "cpumask_isolated"),
        "queues": queues,
    },
    "canonical_policy": {
        "env_or_harness_controls_workqueue_affinity": [term for term in control_terms if term in source],
        "metrics_record_workqueue_affinity": [term for term in metric_terms if f'"{term}"' in BENCH],
    },
}
assert report["live_policy"]["online_cpus"]
assert report["live_policy"]["global_cpumask"]
assert queues and all(item["cpumask"] for item in queues.values())
assert not report["canonical_policy"]["env_or_harness_controls_workqueue_affinity"]
assert not report["canonical_policy"]["metrics_record_workqueue_affinity"]
out = ROOT / "results/kernel-unbound-workqueue-affinity-policy-audit-20260807.json"
out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
print(out)
