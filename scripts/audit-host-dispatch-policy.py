#!/usr/bin/env python3
"""Audit llama-bench host polling and process-priority policy."""
import json
import os
import re
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BENCH = Path(os.environ.get("LX_LLAMA_BENCH", "/home/frosty40/turbo/worktrees/treebeard-base-control-latest/build-mmadd-decode/bin/llama-bench"))
SOURCES = [ROOT / "env.sh", ROOT / "scripts/bench-serial.sh"]
CONTROLS = {"poll": r"(?:--poll\b|\bLX_POLL\b)", "priority": r"(?:--prio\b|\bLX_PRIO\b)"}

def mentions(text):
    return {name: len(re.findall(pattern, text)) for name, pattern in CONTROLS.items()}

help_text = subprocess.run([str(BENCH), "--help"], text=True, stdout=subprocess.PIPE,
                           stderr=subprocess.STDOUT, check=True).stdout
active = {str(path.relative_to(ROOT)): mentions(path.read_text()) for path in SOURCES}
artifacts = {name: 0 for name in CONTROLS}
parsed = 0
for path in ROOT.rglob("*.json"):
    if path.name.startswith("host-dispatch-policy-audit-"):
        continue
    try:
        text = path.read_text()
        json.loads(text)
    except (OSError, UnicodeError, json.JSONDecodeError):
        continue
    parsed += 1
    found = mentions(text)
    for name in CONTROLS:
        artifacts[name] += int(found[name] > 0)
report = {
    "binary": str(BENCH),
    "defaults": {
        "poll": int(re.search(r"--poll <0\.\.\.100>\s+\(default: (\d+)\)", help_text).group(1)),
        "priority": int(re.search(r"--prio <-1\|0\|1\|2\|3>\s+process/thread priority \(default: (-?\d+)\)", help_text).group(1)),
    },
    "active_source_mentions": active,
    "environment": {"LX_POLL": os.environ.get("LX_POLL"), "LX_PRIO": os.environ.get("LX_PRIO")},
    "parsed_json_artifacts": parsed,
    "artifacts_with_control": artifacts,
}
out = ROOT / "results" / "host-dispatch-policy-audit-20260807.json"
out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
print(out)
