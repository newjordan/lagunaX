#!/usr/bin/env python3
"""Audit accelerator-device exclusivity and competing-process provenance."""
import json
import pathlib
import subprocess
from datetime import datetime, timezone

root = pathlib.Path(__file__).resolve().parents[1]
bench = (root / "scripts/bench-serial.sh").read_text()
env = (root / "env.sh").read_text()

def run(*args):
    p = subprocess.run(args, text=True, capture_output=True, check=False)
    return {"command": list(args), "returncode": p.returncode,
            "stdout_lines": [x for x in p.stdout.splitlines() if x.strip()],
            "stderr_lines": [x for x in p.stderr.splitlines() if x.strip()]}

render_users = run("fuser", "/dev/dri/renderD128")
processes = run("ps", "-eo", "pid=,comm=,args=")
accelerator_processes = [line.strip() for line in processes["stdout_lines"]
                         if any(term in line.lower() for term in
                                ("llama", "ollama", "vllm", "sglang", "level-zero"))]
report = {
    "direction": "accelerator exclusivity and competing-workload provenance",
    "generated_utc": datetime.now(timezone.utc).isoformat(),
    "device": "/dev/dri/renderD128",
    "device_user_probe": render_users,
    "accelerator_related_processes": accelerator_processes,
    "canonical_policy": {
        "environment_declares_exclusivity": "EXCLUSIVE" in env or "DEVICE_LOCK" in env,
        "harness_checks_device_users": "fuser" in bench or "lsof" in bench,
        "harness_records_device_user_pids": "device_user" in bench.lower() or "accelerator_process" in bench.lower(),
    },
}
report["finding"] = (
    "The canonical benchmark neither verifies accelerator exclusivity nor records device-user PIDs; "
    "the live probe captures current device users and accelerator-related processes."
)
out = root / "results/accelerator-exclusivity-policy-audit-20260807.json"
out.write_text(json.dumps(report, indent=2) + "\n")
assert not report["canonical_policy"]["harness_checks_device_users"]
assert not report["canonical_policy"]["harness_records_device_user_pids"]
print(out)
