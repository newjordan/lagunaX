#!/usr/bin/env python3
"""Audit KV-cache accelerator-offload policy used by the active serial benchmark."""
from __future__ import annotations

import json
import os
import re
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ENV_PATH = ROOT / "env.sh"
BENCH_PATH = ROOT / "scripts" / "bench-serial.sh"
env_text = ENV_PATH.read_text(errors="replace")
bench_text = BENCH_PATH.read_text(errors="replace")
bench_bin = os.environ.get(
    "LX_LLAMA_BENCH",
    "/home/frosty40/turbo/worktrees/treebeard-base-control-latest/build-mmadd-decode/bin/llama-bench",
)
help_text = subprocess.run(
    [bench_bin, "--help"],
    text=True,
    stdout=subprocess.PIPE,
    stderr=subprocess.STDOUT,
    timeout=30,
    check=True,
).stdout
help_line = next((line.strip() for line in help_text.splitlines() if "--no-kv-offload" in line), None)
default_match = re.search(r"--no-kv-offload[^\n]*\(default:\s*([^\)]+)\)", help_text)
option_pattern = re.compile(r"(?:^|\s)(?:-nkvo|--no-kv-offload)(?:\s|=)", re.MULTILINE)
source_overrides = [
    path
    for path, text in (("env.sh", env_text), ("scripts/bench-serial.sh", bench_text))
    if option_pattern.search(text)
]
environment_overrides = {
    name: os.environ[name]
    for name in ("NKVO", "NO_KV_OFFLOAD", "LLAMA_ARG_NO_KV_OFFLOAD")
    if name in os.environ
}
artifact_mentions = []
parsed_artifacts = 0
for path in sorted(ROOT.glob("**/*.json")):
    if path == Path(__file__):
        continue
    try:
        payload = json.loads(path.read_text(errors="replace"))
    except (OSError, json.JSONDecodeError):
        continue
    parsed_artifacts += 1
    serialized = json.dumps(payload, sort_keys=True).lower()
    if any(token in serialized for token in ("no-kv-offload", "no_kv_offload", '"nkvo"')):
        artifact_mentions.append(str(path.relative_to(ROOT)))

executable_default = int(default_match.group(1)) if default_match else None
explicit_override = bool(source_overrides or environment_overrides)
report = {
    "policy": "kv_cache_accelerator_offload",
    "llama_bench": bench_bin,
    "help_line": help_line,
    "executable_no_kv_offload_default": executable_default,
    "source_overrides": source_overrides,
    "environment_overrides": environment_overrides,
    "active_explicit_override": explicit_override,
    "effective_no_kv_offload": None if explicit_override else executable_default,
    "effective_kv_cache_accelerator_offload": None if explicit_override else executable_default == 0,
    "parsed_json_artifacts": parsed_artifacts,
    "historical_artifact_mentions": artifact_mentions,
}
print(json.dumps(report, indent=2, sort_keys=True))
