#!/usr/bin/env python3
"""Audit llama-bench depth parameter coverage in the active serial workload."""
import json
import re
import subprocess
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ENV = (ROOT / "env.sh").read_text(errors="replace")
BENCH = (ROOT / "scripts/bench-serial.sh").read_text(errors="replace")
OUT = ROOT / "results/depth-workload-coverage-audit-20260807.json"

cmd = f"source {ROOT / 'env.sh'} >/dev/null 2>&1; \"$LX_LLAMA_BENCH\" --help"
help_text = subprocess.run(["bash", "-c", cmd], text=True, capture_output=True, check=True).stdout
help_line = next((line.strip() for line in help_text.splitlines() if "--n-depth" in line), None)
default_match = re.search(r"--n-depth <n>.*?\(default:\s*([^)]+)\)", help_text)

values = []
artifacts = []
parsed = 0
for path in sorted(ROOT.rglob("*.json")):
    if path == OUT:
        continue
    try:
        data = json.loads(path.read_text())
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        continue
    parsed += 1
    found = []
    def walk(node):
        if isinstance(node, dict):
            for key, value in node.items():
                if key == "n_depth":
                    found.append(value)
                walk(value)
        elif isinstance(node, list):
            for value in node:
                walk(value)
    walk(data)
    if found:
        artifacts.append(str(path.relative_to(ROOT)))
        values.extend(found)

source_explicit = bool(re.search(r"(?:--n-depth|N_DEPTH|(?m:^\s*-d(?:\s|$)))", ENV + "\n" + BENCH))
payload = {
    "control": "-d/--n-depth <n>",
    "help_line": help_line,
    "executable_default": int(default_match.group(1)) if default_match else None,
    "active_source_explicit": source_explicit,
    "effective_active_depth": None if source_explicit else (int(default_match.group(1)) if default_match else None),
    "parsed_json_artifacts": parsed,
    "artifacts_with_n_depth": len(artifacts),
    "artifact_paths": artifacts,
    "n_depth_value_counts": dict(sorted(Counter(str(v) for v in values).items())),
}
OUT.write_text(json.dumps(payload, indent=2) + "\n")
print(OUT)
