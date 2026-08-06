#!/usr/bin/env python3
"""Audit KV-cache datatype and device-offload policy used by Laguna benchmarks."""
import json
import os
import re
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
env_text = (ROOT / "env.sh").read_text()
bench_text = (ROOT / "scripts/bench-serial.sh").read_text()

def env_default(name: str):
    match = re.search(rf'export {name}="\$\{{{name}:-([^}}]+)\}}"', env_text)
    return match.group(1) if match else None

binary = Path(os.environ.get("LX_BIN", ROOT / "build-ipex-latest/bin")) / "llama-bench"
help_text = subprocess.run([str(binary), "--help"], text=True, capture_output=True, check=True).stdout

def help_default(flag: str):
    match = re.search(rf'{re.escape(flag)}[^\n]*\(default: ([^)]+)\)', help_text)
    return match.group(1) if match else None

files = records = 0
pairs = {}
for path in ROOT.rglob("*.json"):
    try:
        data = json.loads(path.read_text())
    except Exception:
        continue
    files += 1
    stack = [data]
    while stack:
        item = stack.pop()
        if isinstance(item, dict):
            stack.extend(item.values())
            k = item.get("ctk") or item.get("cache_type_k")
            v = item.get("ctv") or item.get("cache_type_v")
            if k is not None or v is not None:
                records += 1
                key = f"{k}/{v}"
                pairs[key] = pairs.get(key, 0) + 1
        elif isinstance(item, list):
            stack.extend(item)

out = {
    "angle": "kv-cache-datatype-and-offload-policy",
    "executable_defaults": {
        "cache_type_k": help_default("--cache-type-k"),
        "cache_type_v": help_default("--cache-type-v"),
        "no_kv_offload": help_default("--no-kv-offload"),
    },
    "active_policy": {
        "cache_type_k": env_default("CTK"),
        "cache_type_v": env_default("CTV"),
        "kv_offload_enabled": "--no-kv-offload" not in bench_text and "-nkvo" not in bench_text,
        "harness_passes_cache_types": '-ctk "$CTK"' in bench_text and '-ctv "$CTV"' in bench_text,
    },
    "historical_json": {"files_parsed": files, "records": records, "type_pairs": pairs},
}
print(json.dumps(out, indent=2, sort_keys=True))
