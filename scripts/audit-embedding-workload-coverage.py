#!/usr/bin/env python3
"""Audit embedding-mode coverage in the active serial Laguna workload."""
import json
import re
import subprocess
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ENV_PATH = ROOT / "env.sh"
BENCH_PATH = ROOT / "scripts/bench-serial.sh"
OUT = ROOT / "results/embedding-workload-coverage-audit-20260807.json"
ENV = ENV_PATH.read_text(errors="replace")
BENCH = BENCH_PATH.read_text(errors="replace")

cmd = f"source {ENV_PATH} >/dev/null 2>&1; \"$LX_LLAMA_BENCH\" --help"
help_text = subprocess.run(["bash", "-c", cmd], text=True, capture_output=True, check=True).stdout
help_line = next((line.strip() for line in help_text.splitlines() if "--embeddings" in line), None)
default_match = re.search(r"--embeddings <0\|1>.*?\(default:\s*([01])\)", help_text)

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
                if key in {"embeddings", "embedding", "embd"}:
                    found.append(value)
                walk(value)
        elif isinstance(node, list):
            for value in node:
                walk(value)
    walk(data)
    if found:
        artifacts.append(str(path.relative_to(ROOT)))
        values.extend(found)

source_explicit = bool(re.search(r"(?:--embeddings|(?m:^|\s)-embd(?:\s|$)|EMBEDDINGS)", ENV + "\n" + BENCH))
payload = {
    "control": "-embd/--embeddings <0|1>",
    "help_line": help_line,
    "executable_default": int(default_match.group(1)) if default_match else None,
    "active_source_explicit": source_explicit,
    "effective_active_embeddings": None if source_explicit else (int(default_match.group(1)) if default_match else None),
    "parsed_json_artifacts": parsed,
    "artifacts_with_embedding_mode": len(artifacts),
    "artifact_paths": artifacts,
    "embedding_value_counts": dict(sorted(Counter(str(v) for v in values).items())),
}
assert help_line and payload["executable_default"] == 0
assert not source_explicit
assert parsed > 0
OUT.write_text(json.dumps(payload, indent=2) + "\n")
print(OUT)
