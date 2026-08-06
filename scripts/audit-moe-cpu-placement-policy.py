#!/usr/bin/env python3
"""Audit MoE expert CPU placement policy and provenance for Laguna."""
import json, os, re, subprocess
from pathlib import Path

root = Path(__file__).resolve().parents[1]
env = (root / "env.sh").read_text()
bench = (root / "scripts/bench-serial.sh").read_text()
cmd = ["bash", "-lc", f"source {root/'env.sh'} >/dev/null 2>&1; printf '%s\\n' \"$LX_LLAMA_BENCH\"; \"$LX_LLAMA_BENCH\" --help 2>&1 || true"]
probe = subprocess.check_output(cmd, text=True)
binary, help_text = probe.split("\n", 1)
match = re.search(r"-ncmoe, --n-cpu-moe <n>\s+\(default: ([^)]+)\)", help_text)
if not match:
    raise SystemExit("active llama-bench does not expose --n-cpu-moe as expected")
source = env + "\n" + bench
configured = bool(re.search(r"(?:-ncmoe|--n-cpu-moe)", source))
recorded = bool(re.search(r"(?:n_cpu_moe|cpu_moe|n-cpu-moe)", bench, re.I))
payload = {
    "angle": "MoE expert CPU placement",
    "binary": binary,
    "live_contract": {"flag": "--n-cpu-moe", "default": int(match.group(1))},
    "laguna": {"explicitly_configured": configured, "recorded_in_metrics": recorded},
    "interpretation": "Zero keeps all MoE expert layers out of explicit CPU placement.",
}
assert payload["live_contract"]["default"] == 0
assert not configured and not recorded
out = root / "results" / "moe-cpu-placement-policy-audit-20260807.json"
out.write_text(json.dumps(payload, indent=2) + "\n")
print(out)
