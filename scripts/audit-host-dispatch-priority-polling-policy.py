#!/usr/bin/env python3
"""Audit process-priority and worker-polling policy in canonical Laguna benchmarks."""
import json
import re
import subprocess
from pathlib import Path

root = Path(__file__).resolve().parents[1]
env_text = (root / "env.sh").read_text()
bench_text = (root / "scripts/bench-serial.sh").read_text()
source = env_text + "\n" + bench_text

probe = subprocess.run(
    ["bash", "-lc", 'source ./env.sh; "$LX_LLAMA_BENCH" --help 2>&1'],
    cwd=root, text=True, capture_output=True, check=True,
).stdout
prio = re.search(r"--prio <([^>]+)>.*?\(default: ([^)]+)\)", probe)
poll = re.search(r"--poll <([^>]+)>.*?\(default: ([^)]+)\)", probe)
assert prio and poll, "llama-bench priority/poll controls not found"

artifact = {
    "audit": "host-dispatch-priority-polling-policy",
    "executable_contract": {
        "priority": {"accepted": prio.group(1), "default": int(prio.group(2))},
        "poll": {"accepted": poll.group(1), "default": int(poll.group(2))},
    },
    "canonical_configured": {
        "priority": bool(re.search(r"(?:--prio|\bPRIO=)", source)),
        "poll": bool(re.search(r"(?:--poll|\bPOLL=)", source)),
    },
    "metrics_recorded": {
        "priority": bool(re.search(r'["\'](?:prio|priority)["\']\s*:', bench_text)),
        "poll": bool(re.search(r'["\']poll["\']\s*:', bench_text)),
    },
}
out = root / "results/host-dispatch-priority-polling-policy-audit-20260807.json"
out.write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n")
print(out)
