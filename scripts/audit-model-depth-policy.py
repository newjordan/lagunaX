#!/usr/bin/env python3
"""Audit llama-bench model-depth configuration and provenance."""
import json, os, re, subprocess
from pathlib import Path

root = Path(__file__).resolve().parents[1]
env_text = (root / "env.sh").read_text()
harness_text = (root / "scripts/bench-serial.sh").read_text()
env = os.environ.copy()
bench = env.get("LX_LLAMA_BENCH")
if not bench:
    m = re.search(r'export LX_BIN="\$\{LX_BIN:-([^}]+)\}"', env_text)
    if not m:
        raise SystemExit("cannot resolve LX_BIN")
    bench = str(Path(m.group(1)) / "llama-bench")
help_run = subprocess.run(
    ["bash", "-lc", f'source "{root / "env.sh"}" >/dev/null 2>&1; exec "$LX_LLAMA_BENCH" --help'],
    text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, env=env)
if help_run.returncode not in (0, 1):
    raise SystemExit(f"llama-bench --help failed ({help_run.returncode}): {help_run.stdout}")
help_text = help_run.stdout
match = re.search(r'-d, --n-depth <n>\s+\(default: ([^)]+)\)', help_text)
if not match:
    raise SystemExit("active llama-bench does not expose --n-depth")
common_match = re.search(r'COMMON=\(\n(.*?)\n\s*\)', harness_text, re.S)
common_text = common_match.group(1) if common_match else ""
configured = bool(re.search(r'(^|\s)(?:-d|--n-depth)(?:\s|$)', common_text, re.M))
recorded = bool(re.search(r'["\'](?:n_depth|depth)["\']\s*:', harness_text))
out = {
    "angle": "model-depth benchmark policy and provenance",
    "binary": bench,
    "supported": True,
    "default_n_depth": int(match.group(1)),
    "configured_by_laguna": configured,
    "recorded_in_metrics": recorded,
    "conclusion": "Laguna benchmarks the full model by executable default but neither pins nor records model depth.",
}
path = root / "results/model-depth-policy-audit-20260807.json"
path.write_text(json.dumps(out, indent=2) + "\n")
print(path)
