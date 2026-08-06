#!/usr/bin/env python3
"""Audit llama-bench process/thread priority policy and Laguna provenance."""
import json, os, re, subprocess
from datetime import datetime, timezone
from pathlib import Path

root = Path(__file__).resolve().parents[1]
env_text = (root / "env.sh").read_text()
bench_text = (root / "scripts/bench-serial.sh").read_text()
bin_path = Path(os.environ.get("LX_LLAMA_BENCH", "/home/frosty40/turbo/worktrees/treebeard-base-control-latest/build-mmadd-decode/bin/llama-bench"))
if not bin_path.is_file():
    candidates = sorted(root.glob("baseline/**/llama-bench"))
    if not candidates:
        raise SystemExit("unable to resolve active llama-bench")
    bin_path = candidates[-1]
help_text = subprocess.run([str(bin_path), "--help"], text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, check=True).stdout
match = re.search(r"--prio <-1\|0\|1\|2\|3>\s+process/thread priority \(default: (-?\d+)\)", help_text)
if not match:
    raise SystemExit("active llama-bench did not expose expected --prio contract")
configured = bool(re.search(r"(?:^|\s)--prio(?:\s|$)", bench_text)) or bool(re.search(r"^(?:export )?(?:PRIO|LX_PRIO)=", env_text, re.M))
recorded = bool(re.search(r'["\'](?:prio|priority|thread_priority)["\']\s*:', bench_text))
payload = {
    "collected_utc": datetime.now(timezone.utc).isoformat(),
    "executable": str(bin_path),
    "live_contract": {"option": "--prio", "allowed": [-1, 0, 1, 2, 3], "default": int(match.group(1))},
    "laguna": {"configured": configured, "recorded_in_metrics": recorded},
}
out = root / "results/process-thread-priority-policy-audit-20260807.json"
out.write_text(json.dumps(payload, indent=2) + "\n")
assert payload["live_contract"]["default"] == 0
assert not configured and not recorded
print(out)
