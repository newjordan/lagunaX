#!/usr/bin/env python3
"""Audit Laguna model load-mode configuration and provenance."""
import json, os, re, subprocess
from pathlib import Path

root = Path(__file__).resolve().parents[1]
env = (root / "env.sh").read_text()
bench = (root / "scripts/bench-serial.sh").read_text()
cmd = f"source {root/'env.sh'}; \"$LX_LLAMA_BENCH\" --help"
help_text = subprocess.run(["bash", "-lc", cmd], text=True, capture_output=True, check=True).stdout
match = re.search(r"--load-mode <([^>]+)> \(default: ([^)]+)\)", help_text)
if not match:
    raise SystemExit("live llama-bench does not expose the expected --load-mode contract")
configured = bool(re.search(r"(?:--load-mode|-lm)(?:\s|\")", bench))
recorded = "load_mode" in bench
artifact = {
    "executable": os.environ.get("LX_LLAMA_BENCH") or "resolved through env.sh",
    "supported_modes": match.group(1).split("|"),
    "default": match.group(2),
    "canonical_harness_configures_load_mode": configured,
    "metrics_record_load_mode": recorded,
    "env_mentions_load_mode": "LOAD_MODE" in env or "load-mode" in env,
}
out = root / "results/load-mode-policy-audit-20260807.json"
out.write_text(json.dumps(artifact, indent=2) + "\n")
print(out)
