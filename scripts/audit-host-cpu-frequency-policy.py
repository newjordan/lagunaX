#!/usr/bin/env python3
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ENV = (ROOT / "env.sh").read_text()
BENCH = (ROOT / "scripts/bench-serial.sh").read_text()
CPU = Path("/sys/devices/system/cpu")

def read(path):
    p = Path(path)
    return p.read_text().strip() if p.exists() else None

policies = []
for p in sorted((CPU / "cpufreq").glob("policy*"), key=lambda x: int(x.name[6:])):
    policies.append({
        "policy": p.name,
        "affected_cpus": read(p / "affected_cpus"),
        "driver": read(p / "scaling_driver"),
        "governor": read(p / "scaling_governor"),
        "available_governors": read(p / "scaling_available_governors"),
        "min_khz": int(read(p / "scaling_min_freq") or 0),
        "max_khz": int(read(p / "scaling_max_freq") or 0),
        "cpuinfo_max_khz": int(read(p / "cpuinfo_max_freq") or 0),
        "energy_performance_preference": read(p / "energy_performance_preference"),
    })
text = ENV + "\n" + BENCH
needles = ["scaling_governor", "scaling_min_freq", "scaling_max_freq", "energy_performance_preference", "cpupower"]
out = {
    "schema": "laguna.host-cpu-frequency-policy-audit.v1",
    "policy_count": len(policies),
    "policies": policies,
    "boost": read(CPU / "cpufreq/boost"),
    "intel_pstate_no_turbo": read("/sys/devices/system/cpu/intel_pstate/no_turbo"),
    "laguna_frequency_controls": {n: n in text for n in needles},
    "laguna_records_effective_policy": any(n in BENCH for n in needles),
}
print(json.dumps(out, indent=2))
