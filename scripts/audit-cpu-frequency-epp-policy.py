#!/usr/bin/env python3
"""Audit live CPU frequency/EPP policy and Laguna provenance."""
import json
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ENV = (ROOT / "env.sh").read_text(errors="replace")
BENCH = (ROOT / "scripts/bench-serial.sh").read_text(errors="replace")


def read(path):
    try:
        return path.read_text(errors="replace").strip()
    except OSError:
        return None


cpufreq = Path("/sys/devices/system/cpu/cpufreq")
policies = sorted(cpufreq.glob("policy[0-9]*"), key=lambda p: int(p.name[6:]))
fields = (
    "affected_cpus", "scaling_driver", "scaling_governor", "scaling_min_freq",
    "scaling_max_freq", "cpuinfo_min_freq", "cpuinfo_max_freq",
    "energy_performance_preference", "energy_performance_available_preferences", "boost",
)
live = {p.name: {field: read(p / field) for field in fields} for p in policies}
source = ENV + "\n" + BENCH
control_terms = ("scaling_governor", "scaling_min_freq", "scaling_max_freq", "energy_performance_preference", "CPU_FREQ_GOVERNOR")
metric_terms = ("scaling_driver", "scaling_governor", "scaling_min_freq", "scaling_max_freq", "energy_performance_preference", "cpu_boost")
report = {
    "generated_utc": datetime.now(timezone.utc).isoformat(),
    "angle": "cpu-frequency-epp-policy",
    "live_policy": {"policy_count": len(policies), "policies": live},
    "canonical_policy": {
        "env_or_harness_controls_frequency_policy": [t for t in control_terms if t in source],
        "metrics_record_frequency_policy": [t for t in metric_terms if f'"{t}"' in BENCH],
    },
}
assert policies and all(v["scaling_driver"] and v["scaling_governor"] for v in live.values())
assert not report["canonical_policy"]["env_or_harness_controls_frequency_policy"]
assert not report["canonical_policy"]["metrics_record_frequency_policy"]
out = ROOT / "results/cpu-frequency-epp-policy-audit-20260807.json"
out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
print(out)
