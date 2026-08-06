#!/usr/bin/env python3
"""Audit the active KV-cache data-type policy and its benchmark provenance."""
import json
import os
import re
import subprocess
from pathlib import Path

root = Path(__file__).resolve().parents[1]
env_text = (root / "env.sh").read_text()
bench_text = (root / "scripts/bench-serial.sh").read_text()

shell = "source ./env.sh >/dev/null && printf '%s\\n' \"$LX_LLAMA_BENCH\""
binary = subprocess.check_output(["bash", "-c", shell], cwd=root, text=True).strip()
help_text = subprocess.run(
    ["bash", "-c", 'source ./env.sh >/dev/null && "$LX_LLAMA_BENCH" --help'],
    cwd=root,
    text=True,
    stdout=subprocess.PIPE,
    stderr=subprocess.STDOUT,
    check=False,
).stdout
if "--cache-type-k" not in help_text:
    raise RuntimeError("active benchmark help unavailable or lacks KV-cache controls")

def default(flag: str) -> str:
    match = re.search(rf"{re.escape(flag)}[^\n]*\(default: ([^)]+)\)", help_text)
    if not match:
        raise RuntimeError(f"missing help default for {flag}")
    return match.group(1)

def env_default(name: str) -> str:
    match = re.search(rf'^export {name}="\$\{{{name}:-([^}}]+)\}}"', env_text, re.M)
    if not match:
        raise RuntimeError(f"missing env default for {name}")
    return match.group(1)

json_files = list((root / "results").rglob("*.json"))
mentions = 0
for path in json_files:
    try:
        text = path.read_text(errors="replace")
    except OSError:
        continue
    if any(term in text for term in ('"ctk"', '"ctv"', 'cache-type-k', 'cache-type-v')):
        mentions += 1

result = {
    "audit": "kv-cache-type-policy",
    "binary": binary,
    "executable_defaults": {
        "cache_type_k": default("--cache-type-k"),
        "cache_type_v": default("--cache-type-v"),
    },
    "environment_defaults": {
        "CTK": env_default("CTK"),
        "CTV": env_default("CTV"),
    },
    "serial_harness": {
        "passes_ctk": '-ctk "$CTK"' in bench_text,
        "passes_ctv": '-ctv "$CTV"' in bench_text,
        "records_ctk": '"ctk": "$CTK"' in bench_text,
        "records_ctv": '"ctv": "$CTV"' in bench_text,
    },
    "history": {"json_files_scanned": len(json_files), "policy_mentions": mentions},
}
result["effective_policy"] = {
    "cache_type_k": result["environment_defaults"]["CTK"],
    "cache_type_v": result["environment_defaults"]["CTV"],
    "quantized": any(v != "f16" for v in result["environment_defaults"].values()),
}

out = root / "results" / "kv-cache-type-policy-audit-20260807.json"
out.write_text(json.dumps(result, indent=2) + "\n")
print(out)
