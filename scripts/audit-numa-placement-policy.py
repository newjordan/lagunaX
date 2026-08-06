#!/usr/bin/env python3
"""Audit host NUMA topology and Laguna's NUMA-placement provenance."""
import json
import re
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
HARNESS = ROOT / "scripts" / "bench-serial.sh"
ENV = ROOT / "env.sh"
OUT = ROOT / "results" / "numa-placement-policy-audit-20260807.json"

def run(*args: str) -> str:
    return subprocess.run(args, text=True, check=True, capture_output=True).stdout.strip()

lscpu = json.loads(run("lscpu", "-J"))["lscpu"]
fields = {row["field"].rstrip(":"): row["data"] for row in lscpu}
node_count = int(fields["NUMA node(s)"])
node_cpus = {key: value for key, value in fields.items() if re.fullmatch(r"NUMA node\d+ CPU\(s\)", key)}
policy = subprocess.run(["numactl", "--show"], text=True, capture_output=True, check=True)
source = HARNESS.read_text() + "\n" + ENV.read_text()
controls = ("numactl", "--membind", "--cpunodebind", "--interleave", "--preferred")
configured = {control: control in source for control in controls}
metrics_record_policy = bool(re.search(r'numa|membind|cpunodebind|interleave', HARNESS.read_text(), re.I))
artifact = {
    "audit": "numa-placement-policy",
    "topology": {"node_count": node_count, "node_cpus": node_cpus},
    "effective_process_policy": policy.stdout.strip().splitlines(),
    "laguna": {
        "controls_present": configured,
        "explicit_policy_configured": any(configured.values()),
        "metrics_record_effective_policy": metrics_record_policy,
    },
    "finding": (
        "The host has one NUMA node spanning CPUs 0-31 and the current default policy binds memory "
        "to node 0; Laguna neither explicitly configures nor records NUMA placement."
    ),
}
assert node_count == 1
assert node_cpus == {"NUMA node0 CPU(s)": "0-31"}
assert not artifact["laguna"]["explicit_policy_configured"]
assert not metrics_record_policy
OUT.parent.mkdir(parents=True, exist_ok=True)
OUT.write_text(json.dumps(artifact, indent=2) + "\n")
print(OUT)
