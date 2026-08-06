#!/usr/bin/env python3
"""Audit dynamic userspace IRQ balancing and Laguna benchmark provenance."""
import json
import pathlib
import re
import subprocess
from datetime import datetime, timezone

ROOT = pathlib.Path(__file__).resolve().parents[1]
ENV = (ROOT / "env.sh").read_text()
BENCH = (ROOT / "scripts/bench-serial.sh").read_text()


def run(*args):
    proc = subprocess.run(args, text=True, capture_output=True, check=False)
    return {"returncode": proc.returncode, "stdout": proc.stdout.strip(), "stderr": proc.stderr.strip()}

processes = run("pgrep", "-a", "irqbalance")
service = run("systemctl", "show", "irqbalance.service", "--property=LoadState,ActiveState,SubState,FragmentPath")
config_paths = [pathlib.Path("/etc/default/irqbalance"), pathlib.Path("/etc/sysconfig/irqbalance")]
configs = {str(path): path.read_text(errors="replace") for path in config_paths if path.is_file()}
source = ENV + "\n" + BENCH
terms = ("irqbalance", "IRQBALANCE_BANNED_CPULIST", "IRQBALANCE_BANNED_CPUS")

payload = {
    "generated_utc": datetime.now(timezone.utc).isoformat(),
    "angle": "dynamic-userspace-irq-balancing-policy",
    "live_state": {
        "irqbalance_processes": processes,
        "irqbalance_service": service,
        "configuration": configs,
        "daemon_active": bool(processes["stdout"]),
    },
    "canonical_policy": {
        "env_or_harness_controls_irqbalance": any(re.search(re.escape(term), source, re.I) for term in terms),
        "metrics_record_irqbalance_state": any(term in BENCH for term in terms),
    },
}
out = ROOT / "results/dynamic-userspace-irq-balancing-policy-audit-20260807.json"
out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
print(out)
assert not payload["canonical_policy"]["env_or_harness_controls_irqbalance"]
assert not payload["canonical_policy"]["metrics_record_irqbalance_state"]
